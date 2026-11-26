# ResNet-32 seed-0 sweep on Vast.ai

Upload `cs260d-resnet32-s0.zip` to `/workspace` on a CUDA-enabled PyTorch instance. This bundle contains the CIFAR-10 data, the exact split used for the dense run, and the saved seed-0 reference. Keep the same `runs` directory when restarting or changing settings to reuse compatible results.

## 1. Upload from Windows

These instructions use the separately prepared seed-0 ZIP. Open PowerShell in the folder containing the ZIP and use the SSH port and `root@address` from the same Vast instance connection entry. You can also upload the ZIP using the Jupyter file browser.

```powershell
$vastPort = Read-Host 'Vast SSH port'
$vastLogin = Read-Host 'Vast SSH destination (root@address)'
scp -P $vastPort .\cs260d-resnet32-s0.zip "${vastLogin}:/workspace/"
```

## 2. Prepare the instance once

Use the Linux terminal on the instance. These commands assume the ZIP is in `/workspace`. If you uploaded it alongside `/workspace`, use `/cs260d-resnet32-s0.zip` as the archive path.

```bash
python -m zipfile -e /workspace/cs260d-resnet32-s0.zip /workspace
cd /workspace/cs260d-resnet32-s0
python -m pip install -r requirements-vast.txt
python -c "import torch, torchvision, filelock, matplotlib; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))"
```

Use the CUDA build of PyTorch and matching torchvision provided by the GPU template. The bundled requirements cover NumPy, filelock, and Matplotlib. Matplotlib is used for the final report.

## 3. Launch or resume

Run this block after setup succeeds. Edit the two decimal arguments at the top for a later sweep. A fraction of `0.25` selects 25% of the training images during discovery. A sparsity of `0.90` removes 90% of the eligible weights. The explicit arguments override the values in the JSON. For the same settings, rerun this launch block after an interruption.

```bash
cd /workspace/cs260d-resnet32-s0 || exit 1
fraction=0.25
sparsity=0.90
log="sweep-f${fraction}-s${sparsity}.log"
nohup python -u run_experiments.py --config resnet32_s0_full_sweep.json \
  --subset-fraction "$fraction" --sparsity "$sparsity" --device cuda \
  >> "$log" 2>&1 < /dev/null &
echo "PID $! | log $log"
```

With one fraction and one sparsity, the planner prints a count of one dense baseline, 18 subsets and 77 masks/final runs. The first configuration ranks weights from the bundled epoch-200 baseline to build the full-data one-shot magnitude mask. On a fresh instance the log prints `Full-data magnitude uses epoch-200.pt (70,400 reference updates, 0 new discovery updates).` Sparse training then starts from epoch 0 and prints one line per epoch. The runner reuses existing masks and final results in the same `runs` directory, including controls completed by the earlier runner. Separate instances can share a completed control by copying its matching `runs/masks/<mask-id>/` and `runs/final/<run-id>/` directories.

## 4. Monitor and collect results

From any new Linux terminal on the same instance, use the log name for your chosen arguments.

```bash
cd /workspace/cs260d-resnet32-s0
log=sweep-f0.25-s0.90.log
tail -n 12 "$log"
grep -E '/runs/final/[^/]+/validation\.json$' "$log" | sort -u | wc -l
```

For the first sweep the second command reaches 77 distinct completed or reused final reports. After all 77, the runner writes `runs.csv`, `summary.csv`, `accuracy-vs-time.png`, and `execution.json` under `runs/sweeps/<sweep-id>/`, then prints that directory as its last line. An interrupted final training job resumes from its latest five-epoch checkpoint. An interrupted mask discovery restarts that mask. Completed subsets, masks and final reports are reused.

After the final sweep directory is printed, archive the outputs on the instance:

```bash
cd /workspace/cs260d-resnet32-s0
python -m zipfile -c /workspace/resnet32-s0-f0.25-s0.90-results.zip runs sweep-f0.25-s0.90.log
```

Then download that ZIP from Windows PowerShell, using the same connection values as in step 1:

```powershell
scp -P $vastPort "${vastLogin}:/workspace/resnet32-s0-f0.25-s0.90-results.zip" .
```

Keep the results ZIP locally before destroying the instance. `runs/subsets/` contains each exact list of image IDs, `runs/masks/` contains each weight mask, `runs/final/` contains per-class validation results and final checkpoints, and `runs/sweeps/` contains the comparisons. If you reuse this root for several sweeps, the archive includes all accumulated runs. Give each download a distinct name.

## Experiments and budgets

- ResNet-32, seed 0, split seed 260, batch size 128. The reference epoch is 10 of the completed 200-epoch dense schedule.
- At 25%, each class contributes 1,125 training images. Random, feature clustering and gradient clustering give three subsets. Five scoring rules (loss, average margin, EL2N, gradient norm and forgetting) each select easy, hard and mixed examples, giving 15 additional subsets.
- Magnitude, eight-round IMP, SNIP and GraSP each use the 18 subsets plus one full-data control. SynFlow uses one synthetic-input control per sparsity. This gives 4 x 19 + 1 = 77 sparse training runs at the requested settings.
- Full-data one-shot magnitude uses the baseline’s 70,400 updates. Its cost record attributes those updates and their original time to the reference. Discovery records the time to construct the mask and zero new optimizer updates. A different full-data update budget runs fresh SGD on its own learning-rate schedule.
- Subset magnitude and subset IMP use 14,080 discovery updates. Full-data IMP uses 70,400 discovery updates. IMP divides the total across eight rounds, giving 1,760 updates per subset round and 8,800 per full-data round.
- SNIP and GraSP use up to 256 class-balanced scoring examples from the indicated pool. SynFlow uses 100 synthetic scoring iterations. Feature and gradient clustering use 15 clustering iterations.
- Each discovered mask starts its final training from the paired epoch-0 weights and trains on all 45,000 training images for 200 epochs. The first sweep ranks configurations on the 5,000 validation images. Test-set evaluation is reserved for the chosen finalists.

## Bundle contents

- `run_experiments.py`, `baseline.py`, `select_subset.py`, `prune.py`, `train_sparse.py`, `common.py`, `models.py`, `report.py`.
- `resnet32_s0_full_sweep.json`, `requirements-vast.txt`, and this guide.
- `data/cifar-10-batches-py/` with the five training batches, `test_batch`, and `batches.meta`. Torchvision checks the packaged dataset files on load.
- `runs/splits/cifar10-260.json` and `runs/baselines/resnet32-s0-f22257622eb6b83d0d9f/` with epochs 0, 10 and 200 plus `dense-validation.json`.
