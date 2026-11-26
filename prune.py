"""Discover a global, unstructured mask over convolutional and linear weights."""
import argparse
import math
import sys
from dataclasses import asdict

import torch
from torch.nn import functional as F

from baseline import ensure
from common import (Experiment, common_args, digest, from_args, generator, install_mask,
                    integer, lock, optimizer, read, save, set_lr, train_pass)
from models import eligible

METHODS = ("magnitude", "imp", "snip", "grasp", "synflow")


def exact_keep(scores, masks, target):
    """Select the exact global survivor count; resolve cutoff ties in parameter order."""
    values = torch.cat([scores[k].detach()[masks[k]].flatten() for k in masks])
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite pruning scores")
    if target == len(values):
        return masks
    cutoff = values.kthvalue(len(values) - target + 1).values
    remaining = target - int((values > cutoff).sum())
    new = {}
    for key, mask in masks.items():
        score = scores[key].detach()
        keep = mask & (score > cutoff)
        if remaining:
            ties = torch.where((mask & (score == cutoff)).flatten())[0][:remaining]
            keep.flatten()[ties] = True
            remaining -= len(ties)
        new[key] = keep
    return new


def score_examples(ids, labels, count, seed):
    if count < 1:
        raise ValueError("Scoring budget must be positive")
    n = min(count, len(ids))
    groups = []
    # The training and selected partitions have equal class quotas.
    for c in range(10):
        pool = ids[labels[ids] == c]
        order = torch.randperm(len(pool), generator=generator(seed + c))
        groups.append(pool[order[:n // 10 + int(c < n % 10)]])
    result = torch.cat(groups)
    return result[torch.randperm(len(result), generator=generator(seed + 31))]


def data_scores(ctx, model, ids, method):
    # BatchNorm uses batch statistics for the data-based saliency objective.
    model.train()
    params = tuple(model.parameters())
    accumulated = [torch.zeros_like(p) for p in params]
    batch = min(ctx.recipe.batch_size, 64)
    temperature = 200.0 if method == "grasp" else 1.0

    def gradients(group, second=False):
        x, y = ctx.images.batch(group)
        loss = F.cross_entropy(model(x) / temperature, y, reduction="sum") / len(ids)
        return torch.autograd.grad(loss, params, create_graph=second)

    for group in ids.split(batch):
        for total, grad in zip(accumulated, gradients(group)):
            total.add_(grad.detach())
    if method == "grasp":
        hv = [torch.zeros_like(p) for p in params]
        for group in ids.split(batch):
            grads = gradients(group, True)
            dot = sum((g * total).sum() for g, total in zip(grads, accumulated))
            for total, v in zip(hv, torch.autograd.grad(dot, params)):
                total.add_(v.detach())
        accumulated = hv
    by_id = {id(p): g for p, g in zip(params, accumulated)}
    result = {}
    for name, layer in eligible(model).items():
        value = layer.weight.detach() * by_id[id(layer.weight)]
        # GraSP's published removal score is -w*Hg. Here higher always means KEEP.
        result[name] = value.abs() if method == "snip" else value
    return result


def discover(ctx, subset=None, method="magnitude", sparsity=.8, updates=None,
             rounds=8, score_count=256, synflow_iterations=100):
    if method not in METHODS or not 0 <= sparsity < 1 or min(rounds, score_count, synflow_iterations) < 1:
        raise ValueError("Invalid pruning configuration")
    if method == "synflow":
        subset = None
    ids = ctx.train
    if subset:
        if subset["recipe"] != asdict(ctx.recipe) or subset["split_id"] != ctx.split_id:
            raise ValueError("Subset and run recipe/split differ")
        ids = torch.tensor(subset["ids"])
        if (digest(subset["ids"]) != subset["ids_digest"] or not len(ids)
                or len(ids.unique()) != len(ids) or not torch.isin(ids, ctx.train).all()):
            raise ValueError("Invalid subset image IDs")
    tag = subset["id"] if subset else ("data-free" if method == "synflow" else "full")
    spec = {"method": method, "sparsity": sparsity, "subset": tag}
    count = 1
    use_baseline = False
    if method in ("magnitude", "imp"):
        batches_per_epoch = math.ceil(len(ctx.train) / ctx.recipe.batch_size)
        if updates is None:
            updates = (40 if subset else ctx.recipe.baseline_epochs) * batches_per_epoch
        count = rounds if method == "imp" else 1
        if updates < count:
            raise ValueError("Each magnitude-pruning round needs at least one optimizer update")
        spec.update(updates=updates, rounds=count)
        use_baseline = (method == "magnitude" and subset is None
                        and updates == ctx.recipe.baseline_epochs * batches_per_epoch)
    elif method == "synflow":
        count = synflow_iterations
        spec["iterations"] = count
    else:
        spec["score_examples"] = min(len(ids), score_count)
    identity = digest([ctx.key, ctx.split_id, spec])
    path = ctx.root / "masks" / identity / "mask.pt"
    with lock(path):
        if path.exists():
            return path
        ck = read(ensure(ctx, 0))
        if use_baseline:
            init_id = ck["init_id"]
            reference_path = ensure(ctx, ctx.recipe.baseline_epochs)
            ck = read(reference_path)
            if (ck["recipe"] != asdict(ctx.recipe) or ck["split_id"] != ctx.split_id
                    or ck["init_id"] != init_id or ck["epoch"] != ctx.recipe.baseline_epochs
                    or ck["updates"] != updates):
                raise ValueError("Dense checkpoint does not match the full-data magnitude configuration")
            print(f"Full-data magnitude uses {reference_path.name} "
                  f"({updates:,} reference updates, 0 new discovery updates).", file=sys.stderr, flush=True)
        begin = ctx.stamp()
        model = ctx.model()
        model.load_state_dict(ck["model"])
        layers = eligible(model)
        masks = {k: torch.ones_like(v.weight, dtype=torch.bool) for k, v in layers.items()}
        total = sum(m.numel() for m in masks.values())
        goal = max(1, round(total * (1 - sparsity)))
        history = []
        if method == "synflow":
            model.double().eval()
            with torch.no_grad():
                for param in model.parameters():
                    param.abs_()
        for step in range(1, count + 1):
            per_round = 0
            if use_baseline:
                scores = {k: layer.weight.detach().abs() for k, layer in layers.items()}
            elif method in ("magnitude", "imp"):
                model = ctx.model()
                model.load_state_dict(ck["model"])
                install_mask(model, masks)
                opt = optimizer(model)
                per_round = updates // count + int(step <= updates % count)
                done = epoch = 0
                while done < per_round:
                    set_lr(opt, done / per_round)
                    result = train_pass(model, opt, ctx, ids, ctx.recipe.seed + epoch + 1,
                                        max_updates=per_round - done)
                    done += result["updates"]
                    epoch += 1
                scores = {k: v.weight_orig.detach().abs() for k, v in eligible(model).items()}
            elif method == "synflow":
                model.zero_grad(set_to_none=True)
                x = torch.ones(1, 3, 32, 32, dtype=torch.float64, device=ctx.device)
                model(x).sum().backward()
                scores = {k: (v.weight.detach() * v.weight.grad).abs() for k, v in layers.items()}
            else:
                batch_ids = score_examples(ids, ctx.images.labels, score_count, ctx.recipe.seed)
                scores = data_scores(ctx, model, batch_ids, method)
            target = round(total * (goal / total) ** (step / count)) if step < count else goal
            masks = exact_keep(scores, masks, target)
            if method == "synflow":
                with torch.no_grad():
                    for k, v in layers.items():
                        v.weight.mul_(masks[k])
            history.append({"round": step, "survivors": target, "updates": per_round})
        masks = {k: v.cpu() for k, v in masks.items()}
        per_layer = {k: {"eligible": v.numel(), "surviving": int(v.sum())} for k, v in masks.items()}
        surviving = sum(x["surviving"] for x in per_layer.values())
        if surviving != goal:
            raise RuntimeError("Global mask count mismatch")
        costs = {k: subset[k] if subset else 0 for k in
                 ("reference_seconds", "reference_updates", "scoring_seconds", "selection_seconds")}
        if use_baseline:
            costs.update(reference_seconds=ck["seconds"], reference_updates=ck["updates"])
            history[0]["checkpoint"] = reference_path.name
        costs.update(discovery_seconds=ctx.stamp() - begin, discovery_updates=sum(r["updates"] for r in history))
        meta = {"id": identity, "recipe": asdict(ctx.recipe), "split_id": ctx.split_id,
                "init_id": ck["init_id"], "spec": spec, "subset_spec": subset["spec"] if subset else None,
                "hardware": torch.cuda.get_device_name(ctx.device) if ctx.device.type == "cuda" else "cpu",
                "per_layer": per_layer, "rounds": history, "eligible": total, "surviving": surviving,
                "actual_sparsity": 1 - surviving / total, "cost": costs}
        save(path.with_name("summary.json"), meta)
        save(path, {**meta, "masks": masks})
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    common_args(p, inherited=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--subset")
    group.add_argument("--full-data", action="store_true")
    p.add_argument("--method", choices=METHODS, default="magnitude")
    p.add_argument("--sparsity", type=float, default=.8)
    p.add_argument("--discovery-updates", type=integer)
    p.add_argument("--rounds", type=integer, default=8)
    p.add_argument("--score-examples", type=integer, default=256)
    p.add_argument("--synflow-iterations", type=integer, default=100)
    args = p.parse_args()
    if args.method != "synflow" and not (args.subset or args.full_data):
        p.error("Choose --subset or --full-data")
    subset = read(args.subset) if args.subset else None
    recipe = from_args(args, subset["recipe"] if subset else None)
    ctx = Experiment(recipe, args.root, args.data_dir, args.device)
    print(discover(ctx, subset, args.method, args.sparsity, args.discovery_updates,
                   args.rounds, args.score_examples, args.synflow_iterations))


if __name__ == "__main__":
    main()
