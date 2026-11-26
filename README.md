# Can Data Selection Reveal Better Lottery Tickets?

We study whether the choice of training images used during pruning can help uncover a better *lottery ticket* on CIFAR-10. A ticket is a sparse subnetwork paired with the original initialization of its surviving weights. One hypothesis is that iterative magnitude pruning (IMP) finds better tickets from images with high forgetting scores than from a random subset of the same size. To judge which ticket is better, we fix the sparsity and dense initialization, train each masked network on the full training split under the same schedule, and compare validation accuracy.

The four Python entry points train a dense baseline, select discovery images, find a weight mask, and train the fixed mask on the full training partition. The sweep runner connects them and produces paired comparisons for each seed.

## Numerical results

[Download the experiment accuracies (JSON)](results/experiment_accuracies.json) for all 628 saved CIFAR-10 validation measurements. The file includes overall accuracy, correct and evaluated image counts, and accuracy for each of the ten classes. It can be used independently of the notebooks and checkpoints.

- `sparse_runs` contains 616 research pruning results and three one-epoch smoke results. Filter by architecture, seed, subset fraction, sparsity, pruner, selector, and difficulty. Use `sweep_id` together with `run_id` to identify a measurement.
- `dense_baselines` contains six 200-epoch research baselines, two 10-epoch timing benchmarks, and one smoke baseline.
- `sweep_summaries` gives each pruner's mean subset accuracy and the available random, full-data, or data-free controls. Accuracy fields are percentages. Subset fraction and sparsity use the 0–1 scale.

## Run the pipeline

For a fresh clone, [create the project environment](#recreate-the-environment) first. Run the following commands in PowerShell from the repository root.

```powershell
.\.venv\Scripts\python.exe run_experiments.py --dry-run
.\.venv\Scripts\python.exe run_experiments.py --config configs/smoke.json
.\.venv\Scripts\python.exe -m unittest -v test_pipeline
```

The example sweep configurations are in [configs/](configs/). The smoke configuration uses one dense epoch, two discovery updates, and one final epoch on the regular 45,000/5,000 train/validation split. Its accuracy is for checking the pipeline. For a research sweep, edit `DEFAULTS` in `run_experiments.py` or pass JSON overrides with `--config`. The default uses ResNet-32, seeds 0/1/2, three subset rules, full-data controls, one-shot magnitude and eight-round IMP, 80% weight sparsity, and 200 final epochs. Run with `--dry-run` to see the counts. Add `vgg19` to the architectures list for the CIFAR VGG-19 variant with batch normalization and a ten-class linear head.

## Four independent commands

```powershell
.\.venv\Scripts\python.exe baseline.py --arch resnet32 --seed 0 --epochs 200
.\.venv\Scripts\python.exe select_subset.py --arch resnet32 --seed 0 --method feature_cluster --subset-fraction 0.2 --reference-epoch 10
.\.venv\Scripts\python.exe prune.py --subset <subset.json> --method imp --sparsity 0.8 --rounds 8 --discovery-updates 14080
.\.venv\Scripts\python.exe train_sparse.py --mask <mask.pt> --epochs 200
```

Replace the bracketed paths with the paths printed by the preceding command. All stages accept `--root`, `--data-dir`, and `--device auto|cuda|cpu`. Pass the same artifact root to downstream stages. Architecture and seed are inherited from the input artifact on the last two commands. Explicit overrides must match.

**Baseline.** A fixed, stratified split is reused across model seeds. Every dense schedule saves initialization, epochs 5/10/15/20 reached during training, the requested stopping epoch, and the final state. `--until 10` stops at epoch 10 under the 200-epoch schedule and can be resumed. `--epochs 10` creates a separate ten-epoch recipe. Selection retrieves its reference automatically, continuing the matching baseline if needed. Recorded training observations include cumulative signed margin, number of observations, previous correctness, ever-learned status and forgetting count. The same optimizer recipe and deterministic per-epoch batch/augmentation seeds are used for dense and final training. Hardware and software changes can affect floating-point results.

**Selection.** Available methods are `random`, `feature_cluster`, `gradient_cluster`, `loss`, `average_margin`, `el2n`, `gradient_norm`, and `forgetting`. The last five support `--difficulty easy|hard|mixed`. Average margin uses the true-class logit minus the strongest competing logit, observed during reference training. Low margin indicates difficulty. Never-learned images rank as most difficult for forgetting. The mixed rule divides the class quota between the two tails. Features and unaugmented logits are cached by checkpoint. Gradient embeddings are unnormalized true-label classifier-weight gradients; their norm equals the feature norm times EL2N. Clustering is class-wise mini-batch k-means with Euclidean distance and an actual nearest member per occupied cluster. Empty clusters use the farthest unselected members to fill the exact quota. The per-class quota is rounded to the nearest image with a minimum of one. Sampling always excludes validation and test examples.

**Pruning.** Choose `magnitude`, `imp`, `snip`, `grasp`, or `synflow`. Global masks cover convolutional and linear weights. Bias and batch-normalization parameters remain trainable. Specify a subset path or `--full-data` for the data-based methods. The default subset discovery budget is 40 full-data epochs of optimizer updates. Full-data magnitude and IMP default to the dense schedule length. One-shot full-data magnitude ranks weights from the paired final dense checkpoint when the requested update count matches the dense schedule. Its cost record counts the original reference training once and records zero new discovery updates. A different `--discovery-updates` budget trains on its own schedule. IMP divides the total updates across rounds and follows a geometric survivor schedule, ending at the exact integer count. Each round restores epoch 0 and restarts SGD. SNIP and GraSP use a class-balanced scoring sample capped by `--score-examples` (default 256). GraSP uses temperature 200 and a signed Hessian-gradient score. SynFlow uses a synthetic all-ones input and `--synflow-iterations` (default 100). Its cache is shared across selectors.

**Final training.** A mask stores named Boolean tensors, architecture, seed, split, initialization fingerprint, selection provenance and discovery costs. Final training reloads epoch 0 including BatchNorm state, attaches the masks, and starts fresh SGD. Checkpoints are written every five epochs and at completion. Completed training is reused if final test evaluation is requested later. The default reports validation results. Use `--test` on a locked configuration for the held-out test partition.

## Results

- `runs/baselines/` stores reusable checkpoints, dense metrics and learning curves.
- `runs/subsets/` stores image IDs, class counts, selector details and timings. Score arrays or class-local cluster assignments are indexed by the accompanying training IDs.
- `runs/masks/` stores each mask and a readable summary with surviving counts per layer and per round.
- `runs/final/` stores model checkpoints, learning curves, confusion matrices and correct/total counts by class.
- `runs/sweeps/` stores the configuration, execution time, per-run CSV, seed means/sample SDs, and the accuracy-versus-time plot. Accuracy is a fraction; paired differences are percentage points. Comparisons appear when the corresponding control is in the sweep. Full-data comparisons use the configured full-data budget. For a compute-matched full-pool comparison set `full_discovery_updates` equal to the subset update budget.

Standalone cost charges each learned selector for its reference checkpoint and scoring, even when the sweep reused them. The run CSV lists each phase separately. Execution time records actual elapsed time including cache reuse and analysis. Turn on `coverage` and set `coverage_epoch` to measure class-wise nearest-selected distances with a common reference. Diagnostic time is outside the standalone training cost. The report computes mask Jaccard similarity against the same-seed full-data mask when available. SynFlow has no selection overhead.

## Recreate the environment

Use a standard project-local venv with Python 3.12 or 3.13. The tested Windows GPU build uses the following packages.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

To rerun the notebooks, install the analysis dependencies into the same environment and select it as the notebook kernel.

```powershell
.\.venv\Scripts\python.exe -m pip install pandas scipy ipykernel jinja2
```

CIFAR-10 downloads into the ignored `data` folder. If the original host has a connection error, the downloader tries the same archive on Zenodo and validates it with torchvision's published checksums. [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) · [Archive mirror](https://zenodo.org/records/10089977)

Method references include [Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635), [SNIP](https://arxiv.org/abs/1810.02340), [GraSP](https://arxiv.org/abs/2002.07376), [SynFlow](https://arxiv.org/abs/2006.05467), and [EL2N and gradient scores](https://arxiv.org/abs/2107.07075).
