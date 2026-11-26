"""Small regression fixtures for checkpoints, masks, and experiment comparisons."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from baseline import ensure
from common import Experiment, Recipe, read
from models import Classifier, eligible
from prune import METHODS as PRUNERS, discover, exact_keep
from report import mask_overlap
from run_experiments import DEFAULTS, plan
from select_subset import METHODS as SELECTORS, select
from train_sparse import train_sparse


class PipelineTest(unittest.TestCase):
    def setUp(self):
        local = tempfile.TemporaryDirectory()
        self.addCleanup(local.cleanup)
        self.root = Path(local.name)
        rng = np.random.default_rng(8)
        fixture = SimpleNamespace(data=rng.integers(0, 256, (60, 32, 32, 3), dtype=np.uint8),
                                  targets=[i % 10 for i in range(60)])
        mocked = patch.object(Experiment, "dataset", return_value=fixture)
        mocked.start()
        self.addCleanup(mocked.stop)
        self.ctx = Experiment(Recipe(batch_size=10, baseline_epochs=2), self.root, self.root, "cpu")

    def test_checkpoint_resume(self):
        ctx = self.ctx
        ensure(ctx, 1)
        ck = read(ensure(ctx, 2))
        other = Experiment(ctx.recipe, self.root / "direct", self.root, "cpu")
        direct = read(ensure(other, 2))
        self.assertEqual(ck["init_id"], direct["init_id"])
        self.assertTrue(all(torch.equal(t, direct["model"][k]) for k, t in ck["model"].items()))
        self.assertTrue(torch.all(ck["stats"]["observations"][ctx.train] == 2))
        self.assertEqual(int(ck["stats"]["observations"][ctx.val].sum()), 0)

    def test_selectors(self):
        for method in SELECTORS:
            with self.subTest(method=method):
                artifact = read(select(self.ctx, method, .4, 1, "mixed", 2))
                self.assertEqual(artifact["class_counts"], [2] * 10)
                self.assertEqual(len(set(artifact["ids"])), 20)
                self.assertFalse(set(artifact["ids"]).intersection(self.ctx.val.tolist()))
        self.assertEqual(select(self.ctx, "random", .4, 1), select(self.ctx, "random", .4, 2))

    def test_full_magnitude_uses_saved_baseline(self):
        ctx = self.ctx
        dense_path = ensure(ctx, ctx.recipe.baseline_epochs)
        dense = read(dense_path)
        with patch("baseline.train_pass", side_effect=AssertionError("baseline retrained")), \
             patch("prune.train_pass", side_effect=AssertionError("extra discovery training")):
            path = discover(ctx, method="magnitude", sparsity=.9)
        mask = read(path)
        weights = {k: dense["model"][k].abs() for k in eligible(ctx.model())}
        active = {k: torch.ones_like(v, dtype=torch.bool) for k, v in weights.items()}
        target = round(.1 * sum(v.numel() for v in weights.values()))
        expected = exact_keep(weights, active, target)
        self.assertEqual(mask["surviving"], target)
        for k, v in expected.items():
            self.assertTrue(torch.equal(mask["masks"][k], v), k)
        self.assertEqual(mask["cost"]["reference_updates"], dense["updates"])
        self.assertEqual(mask["cost"]["reference_seconds"], dense["seconds"])
        self.assertEqual(mask["cost"]["discovery_updates"], 0)
        self.assertEqual(mask["rounds"], [{"round": 1, "survivors": target, "updates": 0,
                                           "checkpoint": dense_path.name}])
        with patch("prune.ensure", side_effect=AssertionError("cached mask was rebuilt")):
            self.assertEqual(discover(ctx, method="magnitude", sparsity=.9, updates=dense["updates"]), path)

    def test_discovery_trains_for_other_schedules_subsets_and_imp(self):
        ctx = self.ctx
        subset = read(select(ctx, "random", .4))
        full_updates = 10  # 50 training examples, batches of 10, two baseline epochs.
        for method, source, updates in [("magnitude", None, 5), ("magnitude", subset, full_updates),
                                        ("imp", None, full_updates)]:
            with self.subTest(method=method, subset=source is not None, updates=updates):
                mask = read(discover(ctx, source, method, .9, updates=updates, rounds=2))
                self.assertEqual(mask["cost"]["discovery_updates"], updates)
                self.assertNotIn("checkpoint", mask["rounds"][0])
        self.assertFalse(ctx.epoch_path(ctx.recipe.baseline_epochs).exists())

    def test_masks_and_training(self):
        ctx = self.ctx
        source = read(select(ctx, "random", .6))
        for method in PRUNERS:
            with self.subTest(method=method):
                path = discover(ctx, source, method, .8, 2, 2, 20, 2)
                artifact = read(path)
                masks = artifact["masks"]
                count = sum(int(m.sum()) for m in masks.values())
                self.assertEqual(count, round(.2 * artifact["eligible"]))
                self.assertEqual(artifact["surviving"], count)
                self.assertTrue(all(m.dtype == torch.bool for m in masks.values()))
                self.assertEqual(mask_overlap(masks, masks), 1)
                if method == "imp":
                    dest = train_sparse(ctx, path, 1)
                    self.assertEqual(dest, train_sparse(ctx, path, 1))
                    metrics = read(dest)["metrics"]
                    self.assertEqual(sum(map(sum, metrics["confusion"])), len(ctx.val))
                    trained = read(dest.with_name("checkpoint.pt"))["model"]
                    for name, mask in masks.items():
                        self.assertTrue(torch.equal(trained[name + "_mask"].bool(), mask))
                        self.assertTrue(torch.all((trained[name + "_orig"] * mask)[~mask] == 0))

    def test_cutoff_ties_and_sweep_deduplication(self):
        values = {"a": torch.tensor([8., 1., 1.]), "b": torch.ones(4)}
        active = {k: torch.ones_like(v, dtype=torch.bool) for k, v in values.items()}
        active["a"][0] = False
        result = exact_keep(values, active, 3)
        self.assertEqual(sum(int(t.sum()) for t in result.values()), 3)
        self.assertFalse(result["a"][0])
        _, jobs = plan(DEFAULTS | {"reference_epochs": [5, 10, 15], "pruners": ["synflow", "snip"]})
        self.assertEqual(sum(p == "synflow" for _, p, _ in jobs), 1)
        self.assertEqual(sum(s[0] == "random" for s, _, _ in jobs), 1)

    def test_vgg_shape_and_head(self):
        model = Classifier("vgg19").eval()
        with torch.no_grad():
            logits, h = model(torch.zeros(2, 3, 32, 32), return_features=True)
        self.assertEqual(logits.shape, (2, 10))
        self.assertEqual(h.shape, (2, 512))
        self.assertEqual(len(eligible(model)), 17)


if __name__ == "__main__":
    unittest.main()
