"""Shared experiment settings, artifact I/O and CIFAR-10 batches."""
import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import URLError

import torch
from filelock import FileLock
from torch.nn import functional as F
from torch.nn.utils import prune
from torchvision.datasets import CIFAR10

from models import Classifier, eligible

HOME = Path(__file__).resolve().parent
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
VERSION = 1


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
    try:
        if path.suffix == ".pt":
            torch.save(obj, tmp)
        else:
            tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read(path):
    path = Path(path)
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=True)
    return json.loads(path.read_text(encoding="utf-8"))


def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path) + ".lock")


def integer(value, minimum=1):
    v = int(value)
    if v < minimum:
        raise argparse.ArgumentTypeError(f"Expected an integer >= {minimum}")
    return v


def fraction(value):
    v = float(value)
    if not 0 < v <= 1:
        raise argparse.ArgumentTypeError("Expected a fraction in (0, 1]")
    return v


@dataclass(frozen=True)
class Recipe:
    arch: str = "resnet32"
    seed: int = 0
    split_seed: int = 260
    batch_size: int = 128
    baseline_epochs: int = 200
    version: int = VERSION

    def __post_init__(self):
        if self.version != VERSION or self.arch not in ("resnet32", "vgg19"):
            raise ValueError("Unsupported recipe version or architecture")
        if min(self.batch_size, self.baseline_epochs) < 1 or min(self.seed, self.split_seed) < 0:
            raise ValueError("Invalid recipe settings")


def common_args(p, inherited=False):
    p.add_argument("--root", type=Path, default=HOME / "runs")
    p.add_argument("--data-dir", type=Path, default=HOME / "data")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    for key, default, fn in (("arch", "resnet32", str), ("seed", 0, lambda x: integer(x, 0)),
                             ("split-seed", 260, lambda x: integer(x, 0)),
                             ("batch-size", 128, integer), ("baseline-epochs", 200, integer)):
        extra = {"choices": ("resnet32", "vgg19")} if key == "arch" else {}
        p.add_argument("--" + key, type=fn, default=None if inherited else default, **extra)


def from_args(args, inherited=None):
    fields = dict(inherited or asdict(Recipe()))
    for key in ("arch", "seed", "split_seed", "baseline_epochs", "batch_size"):
        value = getattr(args, key, None)
        if value is not None:
            if inherited and fields[key] != value:
                raise ValueError(f"--{key.replace('_', '-')} conflicts with the input artifact")
            fields[key] = value
    return Recipe(**fields)


def generator(seed):
    return torch.Generator().manual_seed(seed)


def partition(labels, seed):
    train, val = [], []
    for c in range(10):
        ids = torch.where(labels == c)[0]
        ids = ids[torch.randperm(len(ids), generator=generator(seed + c))]
        n = max(1, round(len(ids) * 0.1))
        train.extend(ids[n:].tolist())
        val.extend(ids[:n].tolist())
    return {"train": sorted(train), "val": sorted(val)}


def stratified(ids, labels, proportion, seed):
    groups = []
    for c in range(10):
        group = ids[labels[ids] == c]
        count = max(1, round(len(group) * proportion))
        groups.append(group[torch.randperm(len(group), generator=generator(seed + c))[:count]])
    return torch.cat(groups).sort().values


class Images:
    def __init__(self, dataset, device):
        self.labels = torch.as_tensor(dataset.targets, dtype=torch.long)
        self.targets = self.labels.to(device)
        self.images = torch.as_tensor(dataset.data, device=device)
        self.device = device
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)

    def batch(self, ids, rng=None):
        idx = ids.to(self.device)
        x = self.images[idx].permute(0, 3, 1, 2).float().div_(255)
        if rng is not None:
            x = F.pad(x, (4, 4, 4, 4), mode="reflect").permute(0, 2, 3, 1)
            n = len(idx)
            offsets = torch.randint(9, (2, n, 1, 1), generator=rng).to(self.device)
            cols = torch.arange(32, device=self.device).view(1, 1, 32)
            flip = (torch.rand(n, 1, 1, generator=rng) < 0.5).to(self.device)
            cols = torch.where(flip, 31 - cols, cols) + offsets[1]
            rows = torch.arange(32, device=self.device).view(1, 32, 1) + offsets[0]
            x = x[torch.arange(n, device=self.device)[:, None, None], rows, cols].permute(0, 3, 1, 2)
        return (x - self.mean) / self.std, self.targets[idx]


class MirroredCIFAR10(CIFAR10):
    # Identical archive, validated with torchvision's original archive and batch digests.
    url = "https://zenodo.org/records/10089977/files/cifar-10-python.tar.gz"


class Experiment:
    def __init__(self, recipe, root, data_dir, device="auto"):
        self.recipe = recipe
        self.root = Path(root).resolve()
        self.data_dir = Path(data_dir).resolve()
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        if self.device.type == "cpu":
            torch.set_num_threads(min(6, torch.get_num_threads()))
        self.key = digest(asdict(recipe))
        self.base = self.root / "baselines" / f"{recipe.arch}-s{recipe.seed}-{self.key}"
        self.images = Images(self.dataset(True), self.device)
        split_path = self.root / "splits" / f"cifar10-{recipe.split_seed}.json"
        with lock(split_path):
            if not split_path.exists():
                save(split_path, partition(self.images.labels, recipe.split_seed))
        split = read(split_path)
        self.split_id = digest(split)
        self.train = torch.tensor(split["train"])
        self.val = torch.tensor(split["val"])
        self._test = None

    def dataset(self, train):
        with lock(self.data_dir / "cifar10"):
            try:
                return CIFAR10(str(self.data_dir), train=train, download=True)
            except URLError:
                return MirroredCIFAR10(str(self.data_dir), train=train, download=True)

    def test(self):
        if self._test is None:
            self._test = Images(self.dataset(False), self.device)
        return self._test

    def epoch_path(self, epoch):
        return self.base / f"epoch-{epoch:03d}.pt"

    def model(self):
        torch.manual_seed(self.recipe.seed)
        return Classifier(self.recipe.arch).to(self.device)

    def restored(self, epoch=0):
        model = self.model()
        checkpoint = read(self.epoch_path(epoch))
        model.load_state_dict(checkpoint["model"])
        return model

    def stamp(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()


def set_lr(opt, progress):
    value = 0.1 * (1 + math.cos(math.pi * progress)) / 2
    for group in opt.param_groups:
        group["lr"] = value


def optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


def scores(logits, y):
    rivals = logits.detach().scatter(1, y[:, None], -torch.inf).amax(1)
    return logits.detach().gather(1, y[:, None]).flatten() - rivals


def train_pass(model, opt, ctx, ids, seed, max_updates=None, stats=None):
    model.train()
    rng = generator(seed)
    order = ids[torch.randperm(len(ids), generator=rng)]
    loss_sum = correct = seen = steps = 0
    for group in order.split(ctx.recipe.batch_size):
        if max_updates is not None and steps >= max_updates:
            break
        x, y = ctx.images.batch(group, rng)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        hits = logits.detach().argmax(1) == y
        if stats is not None:
            hits_cpu = hits.cpu()
            stats["margin"][group] += scores(logits, y).cpu()
            stats["forgot"][group] += (stats["previous"][group] & ~hits_cpu).int()
            stats["previous"][group] = hits_cpu
            stats["learned"][group] |= hits_cpu
            stats["observations"][group] += 1
        loss_sum += float(loss.detach()) * len(y)
        correct += int(hits.sum())
        seen += len(y)
        steps += 1
    return {"loss": loss_sum / seen, "accuracy": correct / seen, "updates": steps}


@torch.no_grad()
def evaluate(model, images, ids, batch_size):
    model.eval()
    confusion = torch.zeros(100, dtype=torch.long, device=images.device)
    loss = 0
    for group in ids.split(batch_size):
        x, y = images.batch(group)
        logits = model(x)
        loss += float(F.cross_entropy(logits, y, reduction="sum"))
        confusion += torch.bincount(y * 10 + logits.argmax(1), minlength=100)
    matrix = confusion.cpu().reshape(10, 10)
    per_class = {}
    for i, name in enumerate(CLASSES):
        count, correct = int(matrix[i].sum()), int(matrix[i, i])
        per_class[name] = {"correct": correct, "total": count, "accuracy": correct / count if count else None}
    return {"loss": loss / len(ids), "accuracy": int(matrix.diag().sum()) / len(ids),
            "per_class": per_class, "confusion": matrix.tolist()}


def install_mask(model, masks):
    layers = eligible(model)
    if layers.keys() != masks.keys():
        raise ValueError("Mask parameter names differ from those in the model")
    for key, layer in layers.items():
        mask = masks[key]
        if mask.dtype != torch.bool or mask.shape != layer.weight.shape:
            raise ValueError(f"Invalid mask for {key}")
        prune.custom_from_mask(layer, "weight", mask.to(layer.weight.device))
