"""Summarize paired runs, compute diagnostics, and draw accuracy versus compute."""
import csv
import json
import statistics
from collections import defaultdict

import torch

from common import digest, read
from select_subset import embeddings


def coverage(ctx, subset, epoch):
    """Average distance to a selected training example of the same class."""
    h = embeddings(ctx, epoch)["h"]
    labels = ctx.images.labels[ctx.train]
    chosen = torch.isin(ctx.train, torch.tensor(subset["ids"]))
    total = 0.
    for c in range(10):
        points = h[labels == c].to(ctx.device)
        reps = h[(labels == c) & chosen].to(ctx.device)
        for part in points.split(512):
            total += float(torch.cdist(part, reps).min(1).values.sum())
    return total / len(h)


def mask_overlap(a, b):
    intersect = union = 0
    for k, mask in a.items():
        intersect += int((mask & b[k]).sum())
        union += int((mask | b[k]).sum())
    return intersect / union if union else 1.


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(out, results, dense, root, cover):
    rows, groups = [], defaultdict(list)
    controls = {}
    for r in results:
        spec = r["subset_spec"] or {}
        if spec.get("method") == "random" or r["pruning"]["subset"] == "full":
            key = (digest(r["recipe"]), r["epochs"], r["pruning"]["method"], r["pruning"]["sparsity"])
            controls.setdefault(key, []).append(r)
    for r in results:
        recipe, spec, pr = r["recipe"], r["subset_spec"] or {}, r["pruning"]
        key = digest(recipe)
        tag = spec.get("method", pr["subset"])
        acc = r["metrics"]["accuracy"]
        base = dense.get(key)
        peers = controls.get((key, r["epochs"], pr["method"], pr["sparsity"]), [])
        setting = {k: v for k, v in pr.items() if k != "subset"}
        random = next((x for x in peers if (x["subset_spec"] or {}).get("method") == "random"
                       and x["subset_spec"]["fraction"] == spec.get("fraction")
                       and {k: v for k, v in x["pruning"].items() if k != "subset"} == setting), None)
        full = next((x for x in peers if x["pruning"]["subset"] == "full"), None)
        jaccard = None
        if full:
            if full["mask"] == r["mask"]:
                jaccard = 1.
            else:
                a = read(root / "masks" / r["mask"] / "mask.pt")["masks"]
                b = read(root / "masks" / full["mask"] / "mask.pt")["masks"]
                jaccard = mask_overlap(a, b)
        row = {"run": r["run"], "architecture": recipe["arch"], "seed": recipe["seed"],
               "partition": r["partition"], "selector": tag, "fraction": spec.get("fraction"),
               "difficulty": spec.get("difficulty"), "reference_epoch": spec.get("reference_epoch"),
               "pruner": pr["method"], "target_sparsity": pr["sparsity"], "actual_sparsity": r["actual_sparsity"],
               "final_epochs": r["epochs"], "accuracy": acc,
               "delta_dense_pp": 100 * (acc - base["metrics"]["accuracy"]) if base else None,
               "delta_random_pp": 100 * (acc - random["metrics"]["accuracy"]) if random else None,
               "delta_full_pp": 100 * (acc - full["metrics"]["accuracy"]) if full else None,
               "coverage_mean_distance": cover.get(pr["subset"]), "mask_jaccard_full": jaccard,
               "hardware": r["hardware"], **r["cost"]}
        for name, stats in r["metrics"]["per_class"].items():
            row[name + "_accuracy"] = stats["accuracy"]
        rows.append(row)
        group = json.dumps([recipe["arch"], spec or tag, setting, r["epochs"]], sort_keys=True)
        groups[group].append(row)
    write_csv(out / "runs.csv", rows)
    aggregated = []
    for label, group in groups.items():
        summary = {"configuration": label, "seeds": len(group)}
        for field in ("accuracy", "total_seconds", "delta_dense_pp", "delta_random_pp", "delta_full_pp"):
            values = [r[field] for r in group if r[field] is not None]
            summary[field + "_mean"] = statistics.mean(values) if values else None
            summary[field + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        aggregated.append(summary)
    write_csv(out / "summary.csv", aggregated)
    if rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        series = defaultdict(list)
        for r in rows:
            series[(r["architecture"], r["pruner"], r["selector"])].append(r)
        for label, items in series.items():
            ax.scatter([r["total_seconds"] / 60 for r in items],
                       [100 * r["accuracy"] for r in items], label=" / ".join(label), alpha=.8)
        if dense:
            ax.scatter([r["seconds"] / 60 for r in dense.values()],
                       [100 * r["metrics"]["accuracy"] for r in dense.values()],
                       label="Dense baseline", marker="x", color="black")
        ax.set(xlabel="Standalone end-to-end time (minutes)", ylabel=f"{rows[0]['partition'].title()} accuracy (%)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "accuracy-vs-time.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
