"""Choose a fixed class-balanced subset of the CIFAR-10 training partition."""
import argparse
from dataclasses import asdict

import torch
from torch.nn import functional as F

from baseline import ensure
from common import Experiment, common_args, digest, fraction, from_args, generator, integer, lock, read, save, stratified

METHODS = ("random", "feature_cluster", "gradient_cluster", "loss", "average_margin",
           "el2n", "gradient_norm", "forgetting")
SCORES = METHODS[3:]


@torch.no_grad()
def embeddings(ctx, epoch):
    checkpoint = ensure(ctx, epoch)
    path = ctx.base / f"features-{epoch:03d}.pt"
    with lock(path):
        if not path.exists():
            start = ctx.stamp()
            model = ctx.model()
            model.load_state_dict(read(checkpoint)["model"])
            model.eval()
            features, predictions = [], []
            for ids in ctx.train.split(ctx.recipe.batch_size):
                x, _ = ctx.images.batch(ids)
                logits, h = model(x, return_features=True)
                features.append(h.cpu())
                predictions.append(logits.cpu())
            save(path, {"h": torch.cat(features), "logits": torch.cat(predictions),
                        "seconds": ctx.stamp() - start})
    return read(path)


def cluster(vectors, k, rng, iterations):
    """Mini-batch Lloyd updates, then select unique nearest real members."""
    n = len(vectors)
    centers = vectors[torch.randperm(n, generator=rng)[:k]].clone()
    counts = torch.zeros(k, device=vectors.device)
    batch = min(n, max(512, 2 * k))
    for _ in range(iterations):
        ix = torch.randperm(n, generator=rng)[:batch]
        points = vectors[ix]
        group = torch.cdist(points, centers).argmin(1)
        freq = torch.bincount(group, minlength=k)
        sums = torch.zeros_like(centers).index_add_(0, group, points)
        active = freq > 0
        centers[active] = (centers[active] * counts[active, None] + sums[active]) / (counts + freq)[active, None]
        counts += freq
    values, groups = [], []
    for part in vectors.split(512):
        value, group = torch.cdist(part, centers).min(1)
        values.append(value.cpu())
        groups.append(group.cpu())
    values, groups = torch.cat(values), torch.cat(groups)
    representatives = {}
    for idx in torch.argsort(values, stable=True).tolist():
        representatives.setdefault(int(groups[idx]), idx)
    chosen = list(representatives.values())
    if len(chosen) < k:
        used = set(chosen)
        fillers = (idx for idx in torch.argsort(values, descending=True).tolist() if idx not in used)
        chosen.extend(next(fillers) for _ in range(k - len(chosen)))
    return torch.tensor(chosen), groups


def select(ctx, method="random", proportion=.2, epoch=10, difficulty="hard", iterations=15):
    if method not in METHODS or not 0 < proportion <= 1 or difficulty not in ("hard", "easy", "mixed"):
        raise ValueError("Invalid selector configuration")
    if (method != "random" and epoch < 1) or iterations < 1:
        raise ValueError("Reference epochs and clustering iterations must be positive")
    spec = {"method": method, "fraction": proportion}
    if method != "random":
        spec["reference_epoch"] = epoch
    if method in SCORES:
        spec["difficulty"] = difficulty
    if method.endswith("_cluster"):
        spec["iterations"] = iterations
    identity = digest([ctx.key, ctx.split_id, spec])
    path = ctx.root / "subsets" / identity / "subset.json"
    with lock(path):
        if path.exists():
            return path
        ck = read(ensure(ctx, 0 if method == "random" else epoch))
        reference_s, reference_steps = ck["seconds"], ck["updates"]
        stats = ck["stats"]
        del ck
        scoring_s = 0.
        data = None
        if method not in ("random", "average_margin", "forgetting"):
            data = embeddings(ctx, epoch)
            scoring_s = data["seconds"]
        begin = ctx.stamp()
        train = ctx.train
        labels = ctx.images.labels[train]
        details = {}
        if method == "random":
            chosen = stratified(train, ctx.images.labels, proportion, ctx.recipe.seed)
        else:
            if data is not None:
                logits, h = data["logits"], data["h"]
                residual = logits.softmax(1) - F.one_hot(labels, 10)
            if method in SCORES:
                if method == "loss":
                    score = F.cross_entropy(logits, labels, reduction="none")
                elif method == "el2n":
                    score = residual.norm(dim=1)
                elif method == "gradient_norm":
                    score = residual.norm(dim=1) * h.norm(dim=1)
                elif method == "average_margin":
                    score = -stats["margin"][train] / stats["observations"][train].clamp_min(1)
                else:
                    score = stats["forgot"][train].float()
                    score[~stats["learned"][train]] = epoch + 1
                details["hardness"] = score
            selected, assignments = [], torch.empty_like(train)
            for c in range(10):
                ix = torch.where(labels == c)[0]
                k = max(1, round(proportion * len(ix)))
                rng = generator(ctx.recipe.seed + c)
                if method in SCORES:
                    shuffled = ix[torch.randperm(len(ix), generator=rng)]
                    order = shuffled[torch.argsort(score[shuffled], descending=True, stable=True)]
                    if difficulty == "hard":
                        pick = order[:k]
                    elif difficulty == "easy":
                        pick = order[-k:]
                    else:
                        hard = (k + 1) // 2
                        pick = torch.cat((order[:hard], order[len(order) - (k - hard):]))
                else:
                    vectors = h[ix].to(ctx.device)
                    if method == "gradient_cluster":
                        vectors = (residual[ix].to(ctx.device)[:, :, None] * vectors[:, None, :]).flatten(1)
                    reps, assignments[ix] = cluster(vectors, k, rng, iterations)
                    pick = ix[reps]
                selected.append(train[pick])
            chosen = torch.cat(selected).sort().values
            if method.endswith("_cluster"):
                details["class_local_cluster"] = assignments
        select_s = ctx.stamp() - begin
        if details:
            save(path.with_name("scores.pt"), {"train_ids": train, **details})
        ids = chosen.tolist()
        save(path, {"id": identity, "ids_digest": digest(ids), "recipe": asdict(ctx.recipe),
                    "spec": spec, "split_id": ctx.split_id, "ids": ids,
                    "class_counts": ctx.images.labels[chosen].bincount(minlength=10).tolist(),
                    "reference_seconds": reference_s, "reference_updates": reference_steps,
                    "scoring_seconds": scoring_s, "selection_seconds": select_s})
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    common_args(p)
    p.add_argument("--method", choices=METHODS, default="random")
    p.add_argument("--subset-fraction", type=fraction, default=.2)
    p.add_argument("--reference-epoch", "--reference-epochs", type=integer, default=10)
    p.add_argument("--difficulty", choices=("hard", "easy", "mixed"), default="hard")
    p.add_argument("--cluster-iterations", type=integer, default=15)
    args = p.parse_args()
    ctx = Experiment(from_args(args), args.root, args.data_dir, args.device)
    print(select(ctx, args.method, args.subset_fraction, args.reference_epoch, args.difficulty, args.cluster_iterations))


if __name__ == "__main__":
    main()
