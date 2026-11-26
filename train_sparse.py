"""Reset to the paired initialization and train the frozen mask on the full train split."""
import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from baseline import ensure
from common import (Experiment, common_args, digest, evaluate, from_args, install_mask, integer,
                    lock, optimizer, read, save, set_lr, train_pass)


def train_sparse(ctx, mask_path, epochs=200, test=False):
    if epochs < 1:
        raise ValueError("Final epochs must be positive")
    artifact = read(mask_path)
    if artifact["recipe"] != asdict(ctx.recipe) or artifact["split_id"] != ctx.split_id:
        raise ValueError("Mask and final run recipe/split differ")
    init = read(ensure(ctx, 0))
    if init["init_id"] != artifact["init_id"]:
        raise ValueError("Epoch-0 weights do not match mask provenance")
    task_id = digest([artifact["id"], epochs])
    out = ctx.root / "final" / task_id
    ck_path = out / "checkpoint.pt"
    dest = out / ("test.json" if test else "validation.json")
    with lock(out / "training"):
        if dest.exists():
            return dest
        model = ctx.model()
        model.load_state_dict(init["model"])
        install_mask(model, artifact["masks"])
        opt = optimizer(model)
        history, seconds, updates = [], 0., 0
        if ck_path.exists():
            previous = read(ck_path)
            model.load_state_dict(previous["model"])
            opt.load_state_dict(previous["optimizer"])
            history, seconds, updates = previous["history"], previous["seconds"], previous["updates"]
            val = previous["validation"]
            del previous
        for epoch in range(len(history) + 1, epochs + 1):
            start = ctx.stamp()
            set_lr(opt, (epoch - 1) / epochs)
            training = train_pass(model, opt, ctx, ctx.train, ctx.recipe.seed + epoch)
            val = evaluate(model, ctx.images, ctx.val, ctx.recipe.batch_size)
            history.append({"epoch": epoch, "train": training, "val_loss": val["loss"],
                            "val_accuracy": val["accuracy"]})
            seconds += ctx.stamp() - start
            updates += training["updates"]
            print(f"sparse epoch {epoch}/{epochs}: validation {val['accuracy']:.3f}", flush=True)
            if epoch % 5 == 0 or epoch == epochs:
                save(ck_path, {"model": model.state_dict(), "optimizer": opt.state_dict(),
                               "history": history, "seconds": seconds, "updates": updates, "validation": val})
        costs = {**artifact["cost"], "final_seconds": seconds, "final_updates": updates}
        costs["total_seconds"] = sum(costs[k] for k in ("reference_seconds", "scoring_seconds",
                                                       "selection_seconds", "discovery_seconds", "final_seconds"))
        costs["total_updates"] = sum(costs[k] for k in ("reference_updates", "discovery_updates", "final_updates"))
        meta = {"run": task_id, "mask": artifact["id"], "recipe": asdict(ctx.recipe), "epochs": epochs,
                "split_id": ctx.split_id, "init_id": artifact["init_id"], "subset_spec": artifact["subset_spec"],
                "pruning": artifact["spec"], "actual_sparsity": artifact["actual_sparsity"],
                "hardware": artifact["hardware"], "cost": costs, "history": history}
        save(out / "validation.json", {**meta, "partition": "validation", "metrics": val})
        if test:
            images = ctx.test()
            result = evaluate(model, images, torch.arange(len(images.labels)), ctx.recipe.batch_size)
            save(dest, {**meta, "partition": "test", "metrics": result})
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    common_args(p, inherited=True)
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--epochs", type=integer, default=200)
    p.add_argument("--test", action="store_true", help="Also evaluate this fixed configuration on the test partition")
    args = p.parse_args()
    recipe = from_args(args, read(args.mask)["recipe"])
    ctx = Experiment(recipe, args.root, args.data_dir, args.device)
    print(train_sparse(ctx, args.mask, args.epochs, args.test))


if __name__ == "__main__":
    main()
