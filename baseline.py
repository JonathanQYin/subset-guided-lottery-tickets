"""Train or resume a dense reference and save selector statistics."""
import argparse
import hashlib
from dataclasses import asdict, replace

import torch

from common import (Experiment, common_args, evaluate, from_args, integer, lock, optimizer,
                    read, save, set_lr, train_pass)


def ensure(ctx, target):
    if not 0 <= target <= ctx.recipe.baseline_epochs:
        raise ValueError("Reference epoch must be within the baseline schedule")
    dest = ctx.epoch_path(target)
    with lock(ctx.base / "baseline"):
        if dest.exists():
            return dest
        model = ctx.model()
        opt = optimizer(model)
        init = ctx.epoch_path(0)
        if not init.exists():
            n = len(ctx.images.labels)
            stats = {name: torch.zeros(n, dtype=dtype) for name, dtype in (
                ("margin", torch.float32), ("observations", torch.int32), ("forgot", torch.int32),
                ("previous", torch.bool), ("learned", torch.bool))}
            state = model.state_dict()
            fingerprint = hashlib.sha256()
            for name, tensor in state.items():
                fingerprint.update(name.encode())
                fingerprint.update(tensor.detach().cpu().numpy().tobytes())
            save(init, {"recipe": asdict(ctx.recipe), "split_id": ctx.split_id, "epoch": 0,
                        "init_id": fingerprint.hexdigest(), "model": state, "optimizer": opt.state_dict(),
                        "stats": stats, "history": [], "seconds": 0.0, "updates": 0})
        if target == 0:
            return init
        start = max(int(p.stem.split("-")[1]) for p in ctx.base.glob("epoch-*.pt")
                    if int(p.stem.split("-")[1]) < target)
        ck = read(ctx.epoch_path(start))
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        for epoch in range(start + 1, target + 1):
            begin = ctx.stamp()
            set_lr(opt, (epoch - 1) / ctx.recipe.baseline_epochs)
            train = train_pass(model, opt, ctx, ctx.train, ctx.recipe.seed + epoch, stats=ck["stats"])
            val = evaluate(model, ctx.images, ctx.val, ctx.recipe.batch_size)
            ck["history"].append({"epoch": epoch, "train": train, "val_loss": val["loss"],
                                  "val_accuracy": val["accuracy"]})
            ck["seconds"] += ctx.stamp() - begin
            ck["updates"] += train["updates"]
            print(f"dense epoch {epoch}: validation {val['accuracy']:.3f}", flush=True)
            if epoch in (5, 10, 15, 20, target):
                ck.update(epoch=epoch, model=model.state_dict(), optimizer=opt.state_dict(), validation=val)
                save(ctx.epoch_path(epoch), ck)
        return dest


def dense_metrics(ctx, test=False):
    path = ensure(ctx, ctx.recipe.baseline_epochs)
    dest = ctx.base / ("dense-test.json" if test else "dense-validation.json")
    with lock(dest):
        if not dest.exists():
            ck = read(path)
            metrics = ck["validation"]
            if test:
                images = ctx.test()
                model = ctx.model()
                model.load_state_dict(ck["model"])
                metrics = evaluate(model, images, torch.arange(len(images.labels)), ctx.recipe.batch_size)
            save(dest, {"recipe": asdict(ctx.recipe), "metrics": metrics, "history": ck["history"],
                        "seconds": ck["seconds"], "updates": ck["updates"]})
    return read(dest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    common_args(p)
    p.add_argument("--epochs", type=integer, help="Total dense schedule (default 200)")
    p.add_argument("--until", type=lambda s: integer(s, 0), help="Stop early under the same LR schedule")
    p.add_argument("--test", action="store_true", help="Evaluate the completed dense schedule on test data")
    args = p.parse_args()
    recipe = from_args(args)
    if args.epochs is not None:
        recipe = replace(recipe, baseline_epochs=args.epochs)
    target = recipe.baseline_epochs if args.until is None else args.until
    if args.test and target != recipe.baseline_epochs:
        p.error("--test requires the completed baseline")
    ctx = Experiment(recipe, args.root, args.data_dir, args.device)
    print(ensure(ctx, target))
    if target == recipe.baseline_epochs:
        dense_metrics(ctx, args.test)


if __name__ == "__main__":
    main()
