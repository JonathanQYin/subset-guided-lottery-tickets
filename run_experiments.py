"""Plan and run a sweep. Edit DEFAULTS or supply a JSON file with overrides."""
import argparse
import itertools
import json
import math
import time
from pathlib import Path

from baseline import dense_metrics
from common import HOME, Experiment, Recipe, digest, fraction, lock, read, save
from prune import METHODS as PRUNERS, discover
from report import coverage, summarize
from select_subset import METHODS as SELECTORS, SCORES, select
from train_sparse import train_sparse

DEFAULTS = {
    "architectures": ["resnet32"], "seeds": [0, 1, 2], "split_seed": 260,
    "batch_size": 128, "baseline_epochs": 200, "final_epochs": 200,
    "reference_epochs": [10], "selectors": ["random", "feature_cluster", "gradient_cluster"],
    "difficulties": ["hard"], "fractions": [0.2], "cluster_iterations": 15,
    "pruners": ["magnitude", "imp"], "sparsities": [0.8], "rounds": 8,
    "discovery_full_epochs": 40, "discovery_updates": None, "full_discovery_updates": None,
    "score_examples": 256, "synflow_iterations": 100, "include_full": True,
    "coverage": False, "coverage_epoch": 10,
}


def plan(cfg):
    recipes = [Recipe(a, s, cfg["split_seed"], cfg["batch_size"], cfg["baseline_epochs"])
               for a, s in itertools.product(cfg["architectures"], cfg["seeds"])]
    if len(set(recipes)) != len(recipes) or not recipes:
        raise ValueError("Specify unique architecture/seed pairs")
    jobs = []
    for pruner in cfg["pruners"]:
        if pruner not in PRUNERS:
            raise ValueError(f"Unknown pruner {pruner}")
        sources = []
        if pruner == "synflow":
            sources.append(("data-free", None, None, None))
        else:
            if cfg["include_full"]:
                sources.append(("full", None, None, None))
            for method in cfg["selectors"]:
                if method not in SELECTORS:
                    raise ValueError(f"Unknown selector {method}")
                epochs = [None] if method == "random" else cfg["reference_epochs"]
                difficulties = cfg["difficulties"] if method in SCORES else [None]
                for f, e, d in itertools.product(cfg["fractions"], epochs, difficulties):
                    if not 0 < f <= 1 or (e is not None and not 1 <= e <= cfg["baseline_epochs"]):
                        raise ValueError("Invalid subset fraction or reference epoch")
                    if d not in (None, "easy", "hard", "mixed"):
                        raise ValueError(f"Unknown difficulty {d}")
                    sources.append((method, f, e, d))
        for source, sparsity in itertools.product(sources, cfg["sparsities"]):
            if not 0 <= sparsity < 1:
                raise ValueError("Sparsity must be in [0, 1)")
            jobs.append((source, pruner, sparsity))
    if not jobs:
        raise ValueError("Sweep contains no configurations")
    return recipes, list(dict.fromkeys(jobs))


def run(cfg, root, data_dir, device="auto", test=False, dry_run=False):
    recipes, jobs = plan(cfg)
    n = len(recipes)
    unique_subsets = {src for src, _, _ in jobs if src[0] not in ("full", "data-free")}
    print(f"{n} dense baselines | {n * len(unique_subsets)} subsets | "
          f"{n * len(jobs)} masks and final runs", flush=True)
    if dry_run:
        return
    out = Path(root).resolve() / "sweeps" / digest([cfg, test])
    out.mkdir(parents=True, exist_ok=True)
    with lock(out / "sweep"):
        start = time.perf_counter()
        results, dense, cover, events = [], {}, {}, []
        save(out / "config.json", {**cfg, "evaluate_test": test})
        for recipe in recipes:
            ctx = Experiment(recipe, root, data_dir, device)
            dense[ctx.key] = dense_metrics(ctx, test)
            subsets = {}
            for src, pruner, sparsity in jobs:
                method, fraction, epoch, difficulty = src
                subset = None
                if src in unique_subsets:
                    if src not in subsets:
                        path = select(ctx, method, fraction, epoch or 10, difficulty or "hard", cfg["cluster_iterations"])
                        subsets[src] = read(path)
                    subset = subsets[src]
                    if cfg["coverage"] and subset["id"] not in cover:
                        cover[subset["id"]] = coverage(ctx, subset, cfg["coverage_epoch"])
                steps = cfg["full_discovery_updates"] if method == "full" else cfg["discovery_updates"]
                if subset and steps is None:
                    steps = round(cfg["discovery_full_epochs"] * math.ceil(len(ctx.train) / recipe.batch_size))
                before = time.perf_counter()
                mask = discover(ctx, subset, pruner, sparsity, steps, cfg["rounds"],
                                cfg["score_examples"], cfg["synflow_iterations"])
                final_id = digest([mask.parent.name, cfg["final_epochs"]])
                partition = "test" if test else "validation"
                cached = (ctx.root / "final" / final_id / f"{partition}.json").exists()
                result = train_sparse(ctx, mask, cfg["final_epochs"], test)
                results.append(read(result))
                events.append({"run": final_id, "final_cached": cached, "seconds_prune_and_final": time.perf_counter() - before})
                print(result, flush=True)
        summarize(out, results, dense, Path(root).resolve(), cover)
        save(out / "execution.json", {"wall_seconds": time.perf_counter() - start,
                                      "hardware": str(ctx.device), "runs": events})
    print(out, flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--subset-fraction", dest="fractions", type=fraction, nargs="+", metavar="F",
                   help="Subset fraction(s) in (0, 1], overriding the JSON fractions")
    p.add_argument("--sparsity", dest="sparsities", type=float, nargs="+", metavar="S",
                   help="Weight sparsity level(s) in [0, 1), overriding the JSON sparsities")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--test", action="store_true", help="Evaluate fixed configurations on held-out test data")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--root", type=Path, default=HOME / "runs")
    p.add_argument("--data-dir", type=Path, default=HOME / "data")
    args = p.parse_args()
    overrides = json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else {}
    unknown = set(overrides) - DEFAULTS.keys()
    if unknown:
        p.error(f"Unrecognized settings: {sorted(unknown)}")
    if args.fractions is not None:
        overrides["fractions"] = args.fractions
    if args.sparsities is not None:
        overrides["sparsities"] = args.sparsities
    run(DEFAULTS | overrides, args.root, args.data_dir, args.device, args.test, args.dry_run)


if __name__ == "__main__":
    main()
