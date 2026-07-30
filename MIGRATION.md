# Migrating to wtpsplit 3

wtpsplit 3 includes packaging/cleanup and new models - SaT 2.
Existing SaT/WtP (legacy) checkpoints remain compatible.

## Runtime requirements

- Python 3.10 or newer is required.
- `transformers>=5,<6` is required; transformers 4 is no longer supported.
- Training and evaluation modules are source-only research code and are no
  longer included in the PyPI wheel.

## Installation

Core SaT inference:

```bash
pip install wtpsplit
```

ONNX and legacy WtP support are explicit extras:

```bash
pip install "wtpsplit[onnx-cpu]"
pip install "wtpsplit[onnx-gpu]"
pip install "wtpsplit[legacy]"
```

The core install no longer adds pandas, scikit-learn, skops, or
mosestokenizer. `from wtpsplit import WtP` remains compatible after installing
the `legacy` extra.

## API changes

The inference stride now defaults to `64` consistently in `split()` and
`predict_proba()`. Pass `stride=256` explicitly to preserve the old
`predict_proba()` default.

`SaT.segment()` is the new structured API:

```python
result = sat.segment("First sentence. Second sentence.")
result.sentences
result.spans          # half-open source character offsets
result.probabilities  # character-level boundary probabilities
```

`SaT.split()` remains available and delegates to `segment()`.

Batch inputs now return concrete lists from `split()`, `segment()`, and
`predict_proba()`, matching the eager semantics of single-text calls. Pass
`lazy=True` to retain iterator-based evaluation for very large batches.

PyTorch devices can now be selected during construction with `device="cuda"`,
`device="mps"`, or another torch device. `compile=True` enables
`torch.compile`; pass a dictionary to forward compiler options. ONNX devices
remain controlled by `ort_providers`.

The LoRA selector is now named uniformly `domain`:

```python
sat = SaT("sat-3l", domain="ud", language="en")
```

`style_or_domain=` remains as a deprecated keyword alias for this release.
Legacy WtP similarly uses `language=` and `domain=`; `lang_code=` and `style=`
remain deprecated aliases.

## Behavioural changes to segmentation output

Two defaults changed. Both alter the segments returned for the same input, so they are
called out separately from the API renames above.

### Length-constrained segmentation now scores non-boundaries

When `min_length` or `max_length` is set, wtpsplit solves a dynamic program over
segmentations. Through 2.x that objective scored only the positions it *chose* as
boundaries:

```
argmax_C  sum_i [ log prior(len_i) + log p(c_i) ]
```

Because `log p < 0`, every additional boundary only subtracts, so nothing in the objective
ever created a split - only the length prior falling towards zero on long segments did.
Segmentation was length-driven rather than evidence-driven, which produced near
fixed-interval chunking.

The default now includes the complement term, making it a proper Bernoulli likelihood over
all positions:

```
argmax_C  sum_i [ log prior(len_i) + log p(c_i) ]  +  sum_{j not a boundary} log(1 - p_j)
```

Measured on BOUQuET with `sat-12l-sm` and a per-language gaussian prior, mean F1 rises from
**0.834 to 0.896** and the number of languages below 0.90 F1 falls from **267 to 98**.

The `min_length` and `max_length` guarantees are unchanged - only boundary placement
within them moves. To restore the 2.x objective exactly:

```python
sat.split(text, max_length=200, use_negative_evidence=False)
```

This only affects length-constrained calls. Standard, thresholded segmentation is untouched.

### Default thresholds resolve on the checkpoint name, not a substring

The default operating point used to be chosen with `"sm" in model_name_or_path`, tested
against the entire string. Any local path that happened to contain those two letters
selected the `-sm` threshold of `0.25` - ten times the base default - silently:

```python
SaT("/home/smith/sat-3l")        # 2.x: 0.25, because of "smith"
SaT("/tmp/transformers-cache/m") # 2.x: 0.25, because of "transformers"
```

Resolution is now anchored on the final path component and matched against the released
checkpoint names. Unrecognised names fall back to `0.025` **and emit a warning** telling
you to pass `threshold=` explicitly, rather than silently inheriting an operating point
that was tuned for a different model.

If you fine-tuned from a `-sm` checkpoint and renamed it, you will now see that warning and
should pass `threshold=0.25` (or your own calibrated value) explicitly.

## LoRA compatibility

Existing adapters still load and are merged directly into the model.
`merge_lora=False` is no longer supported because AdapterHub currently
requires transformers 4. Use `sat.adapt([...], language=...)` to train a native
LoRA adapter in memory under transformers 5, and `sat.save_adapter(path)` when
an artifact reloadable through `lora_path` is needed.

## Packaging measurements

Measured on macOS arm64 with Python 3.12 and the same transformers 5.14.1 /
torch 2.13 runtime in both environments:

- Wheel size: 148 KiB (`2.2.1`) to 83 KiB (`3.0.0`), a 44% reduction,
  including the dependency-free in-process adaptation module.
- Cold `import wtpsplit` median over seven subprocesses: 4.138 s to 4.226 s.

Cold import is effectively unchanged because importing transformers/torch dominates it; the packaging work primarily removes unnecessary
dependencies and installed source files.
