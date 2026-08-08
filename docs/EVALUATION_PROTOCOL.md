# Stage 1 evaluation

At present, Stage 1 uses the historical SaT intrinsic evaluation for model
selection. Any future extension must be frozen and applied to every arm before
it enters the selection score.

## Intrinsic evaluation

`scripts/evaluate_stage1.py` evaluates a final checkpoint on UD, OPUS100, and
Ersatz from `data/all_data_11_05-all.pth`. The default run uses block size 512,
stride 64, fixed threshold 0.01, up to 10,000 adaptation sentences, and the
historical held-out split rule.

```bash
uv run python scripts/evaluate_stage1.py \
  --model runs/curriculum/stage1_fineweb \
  --output runs/curriculum/evaluation/stage1_fineweb.json
```

The command writes three files:

| File              | Contents                                           |
| ----------------- | -------------------------------------------------- |
| `<name>.json`     | compact corpus and language summary                |
| `<name>.raw.json` | complete intrinsic result                          |
| `<name>.run.json` | command, input paths, output hashes, and timestamp |

The summary reports F1 at the fixed threshold as `@u` and F1 at the
train-mixture tuned threshold as `@t`. The primary score is the mean of each
language's available corpus scores, followed by a mean across languages.
Corpus macros and weak-language results must accompany it; the aggregate alone
can hide failures in OPUS100 or a particular script.

Evaluate final checkpoints rather than an intermediate smoke. Use the same
evaluation packet, dataset list, block size, stride, and adaptation settings
for every arm. Threshold tuning uses training or development material and must
never inspect test labels.
