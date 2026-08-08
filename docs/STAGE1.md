# SaT Stage 1

Stage 1 pretrains the SaT encoder on paragraph boundaries in multilingual web
text. The next experiment compares the historical mC4 source with FineWeb and
tests mmBERT alongside XLM-R. Corpus and backbone are crossed so that the two
effects can be read separately.

Paragraphs remain the main training unit. They match the published SaT setup
and avoid introducing a unit change into the corpus comparison. Document rows
are supported as a follow-up. The completed mC4 unit comparison found almost
no difference between the two choices.

## Experiment

All primary arms use the same 85 languages, per-language character budgets,
200,000 steps, global batch 512, and seed 42.

| Arm                        | Config                                          | Status                    |
| -------------------------- | ----------------------------------------------- | ------------------------- |
| mC4 paragraphs, XLM-R      | `configs/curriculum/stage1_mc4.json`            | complete                  |
| FineWeb paragraphs, XLM-R  | `configs/curriculum/stage1_fineweb.json`        | data and training pending |
| mC4 paragraphs, mmBERT     | `configs/curriculum/stage1_mc4_mmbert.json`     | pending                   |
| FineWeb paragraphs, mmBERT | `configs/curriculum/stage1_fineweb_mmbert.json` | pending                   |

Two XLM-R document arms measure the effect of changing the training unit:

| Arm               | Config                                             | Status   |
| ----------------- | -------------------------------------------------- | -------- |
| mC4 documents     | `configs/curriculum/stage1_mc4_documents.json`     | complete |
| FineWeb documents | `configs/curriculum/stage1_fineweb_documents.json` | pending  |

The XLM-R runs use the historical `punctuation_xlmr_unk.txt` label set. The
mmBERT runs use its tokenizer and `punctuation_extended_mmbert_unk.txt`. Label
counts therefore differ by backbone and must be included in reported run
metadata.

## Pipeline map

The Stage 1 pipeline has eight gates. Each gate leaves an artifact that can be
checked before the next one starts.

| Gate            | Command                                                                         | Evidence produced or checked                                                                |
| --------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Machine         | `scripts/doctor.py`                                                             | JSON report of the checkout, environment, storage, CUDA, and required files                 |
| Plan            | `scripts/dry_run_stage1.py`                                                     | offline report covering sources, budgets, run arms, distributed batch, and scale-out schema |
| Frozen inputs   | `scripts/measure_mc4_test_char_mass.py`, `scripts/build_contamination_index.py` | character-budget manifest and text-free evaluation fingerprint index                        |
| Acquisition     | `scripts/build_stage1.py`                                                       | local Parquet shards and a resumable `metadata.json` receipt                                |
| Corpus gate     | `scripts/validate_stage1.py`                                                    | bounded validation report; exit code 2 on an invalid corpus                                 |
| Run preparation | `scripts/prepare_stage1_runs.py`                                                | four resolved run descriptions and, when requested, rendered distributed configs            |
| Model gate      | `scripts/evaluate_stage1.py`                                                    | raw intrinsic result, compact summary, and hashed run receipt                               |
| Decision        | `configs/curriculum/selections.json`                                           | selected checkpoint and the evidence paths used to choose it                                |

The acquisition gate has a separate bounded-sample route for the 1,870-pair
FineWeb expansion. Its planning, fetching, filtering, and current limitations
are covered in [STAGE1_DATA.md](STAGE1_DATA.md).

## Run order

For a new cluster run, use this order:

1. Choose scratch locations for Hub caches, local corpora, temporary files, and
   model outputs.
2. Install the locked research environment, authenticate, and run the doctor
   in strict GPU mode.
3. Run the offline plan check for the intended GPU count.
4. Make the contamination index available. Rebuild it from the frozen UD and
   BOUQuET inputs when the stored artifact cannot be copied from existing
   research storage.
5. Resolve a bounded FineWeb paragraph build, materialize it in a smoke
   directory, and validate its receipt.
6. Materialize the full matched FineWeb paragraph corpus. Run structural
   validation during construction and hash validation when the final corpus is
   complete or has been copied.
7. Render distributed configs, then run one short XLM-R smoke and one short
   mmBERT smoke. Each smoke must save and reload its checkpoint.
8. Launch the three pending paragraph arms: FineWeb/XLM-R, mC4/mmBERT, and
   FineWeb/mmBERT. Preserve global batch 512 and keep resumable checkpoints.
9. Reload each final checkpoint independently, then evaluate it with the frozen
   intrinsic packet.
10. Compare the four paragraph arms, update `selections.json`, and retain the
    configs, corpus receipts, run plan, logs, checkpoint, evaluation outputs,
    and hashes used for the decision.
11. Run the FineWeb document follow-up only if the paragraph comparison leaves
    a reason to revisit the unit choice.
12. Treat the 1,870-pair FineWeb expansion as a separate run family after its
    materializer and mixture-weighted sampler are finished.

The remaining sections give the commands and acceptance conditions for these
steps.

## Storage layout

All commands assume the repository root as the working directory. The configs
refer to `data/external` and `runs` with relative paths. On a fresh cluster
checkout, those can be symlinks into scratch:

```bash
mkdir -p /scratch/mmsat/huggingface
mkdir -p /scratch/mmsat/external
mkdir -p /scratch/mmsat/runs
mkdir -p /scratch/mmsat/tmp

ln -s /scratch/mmsat/external data/external
ln -s /scratch/mmsat/runs runs

export HF_HOME=/scratch/mmsat/huggingface
export TMPDIR=/scratch/mmsat/tmp
```

If either repository path already exists, keep it and change the relevant
paths in a rendered config instead of replacing it. `data/external`, `runs`,
Hub caches, and temporary files are generated artifacts and stay out of the
handoff commit.

## Required external artifacts

A fresh checkout lacks two large files:

| File | How to obtain it | Required for |
| --- | --- | --- |
| `data/all_data_11_05-all.pth` | copy from the existing SaT research storage | intrinsic evaluation |
| `data/external/mmsat_contamination_index_v1.json.gz` | copy the frozen artifact or rebuild it as described in `STAGE1_DATA.md` | FineWeb materialization |

The historical evaluation packet has SHA-256
`128f87a298aafa80a8388ab2a8f9e8ab76b0e3f2b89ea52bce08c39df90d316d`.
There is currently no public downloader for it. Link or copy it into `data/`
and verify the hash before evaluation. This access dependency should be passed
to a collaborator together with the checkpoint storage location.

The expected contamination-index hash and byte size are recorded in
`data/manifests/mmsat_contamination_index_v1.json`. Verify a copied artifact
against that manifest. A newly rebuilt index is a new data input and needs a
new manifest and corpus receipt.

## Machine preflight

Install the research environment and authenticate with Hugging Face:

```bash
uv sync --locked --group research --extra legacy --extra onnx-cpu
hf auth login
uv run python scripts/doctor.py \
  --output-root /scratch/mmsat/runs \
  --minimum-free-gb 300 \
  --require-gpu \
  --strict
```

`doctor.py` is a machine and checkout check. It reports:

- Python 3.10 or newer and the `datasets`, `pyarrow`, `torch`, and
  `transformers` imports;
- the current Git commit and whether the checkout has local changes;
- presence of the six matched Stage 1 configs, the separate scale-out pilot
  config, their selection file, the frozen matched and scale-out manifests,
  contamination provenance, the UD selection, and
  `data/all_data_11_05-all.pth`;
- whether a Hugging Face token is visible;
- CUDA availability and device count;
- free space on the filesystem that will contain `--output-root`.

With `--strict`, a missing Python requirement, package, required file, requested
GPU, or disk allowance makes the command exit non-zero. Git dirtiness and a
missing Hub token remain diagnostic fields because they are not always errors.
Check `huggingface_token` in the report before fetching, and use `hf auth
whoami` when model or dataset access fails.

The doctor does not download anything, open a remote dataset, estimate GPU
memory, or inspect a built corpus. Those checks belong to the later gates. Put
Hugging Face caches, corpora, temporary files, and run directories on cluster
scratch before starting a full build.

## Offline plan check

The complete plan can be checked without downloading data or loading model
weights:

```bash
uv run python scripts/dry_run_stage1.py \
  --nproc-per-node 4 \
  --smoke-chars-per-language 10000 \
  --write-report runs/preflight/stage1-plan.json
```

`dry_run_stage1.py` resolves the pinned mC4 and FineWeb source mappings for all
85 matched languages, applies a bounded smoke cap without changing the frozen
manifest, checks the required fields for all 1,870 scale-out rows, and builds
the four-arm corpus-by-backbone run plan. It parses all six matched configs and
the separate scale-out pilot config against the current training arguments,
and rejects unknown keys or a distributed layout that cannot preserve global
batch 512.

The report should have `valid: true`, 85 languages under each matched corpus,
four primary arms, and 1,870 FineWeb language-script pairs. A missing local
contamination index is reported as a warning because the offline plan can still
be checked; materializing FineWeb remains blocked until that file exists. The
dry run does not prove that Hub revisions are reachable, that shards fit on
disk, or that a model fits in GPU memory.

Once network access is available, fetch both backbones before reserving a long
GPU job. This checks the model configuration, SaT wrapper resolution, weights,
and tokenizer while populating the shared cache:

```bash
uv run python -c '
from transformers import AutoTokenizer
from wtpsplit.train.backbones import resolve_backbone
for name in ("xlm-roberta-base", "jhu-clsp/mmBERT-base"):
    _, model_class, _ = resolve_backbone(name)
    model_class.from_pretrained(name)
    AutoTokenizer.from_pretrained(name)
    print(f"backbone access: ok: {name}")
'
```

## Build and check the corpora

The pinned `markus583/mC4-TEST` dataset is the source for the main mC4
paragraph arm. FineWeb paragraphs and both document alternatives are local
Parquet builds. Commands, manifests, contamination rules, and the wider
FineWeb plan are described in [STAGE1_DATA.md](STAGE1_DATA.md).

Before training on a local corpus, validate its receipt:

```bash
uv run python scripts/validate_stage1.py \
  data/external/fineweb2-stage1-paragraphs/metadata.json \
  --mapping data/manifests/sat_lang_to_fineweb2_v1.json \
  --require-contamination-filter
```

Use `--verify-hashes` before a full run or after copying a corpus to another
machine. The normal structural pass avoids reading every shard in full.

## Training

A one-GPU launch reads a curriculum config directly:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m wtpsplit.train.train \
  configs/curriculum/stage1_fineweb.json
```

The checked-in configs use per-device batch 64 and accumulation 8. That gives
global batch 512 on one GPU. A four-GPU run needs accumulation 2. Render copies
with the corrected value instead of editing the canonical configs:

```bash
uv run python scripts/prepare_stage1_runs.py \
  --nproc-per-node 4 \
  --render-dir /scratch/mmsat/stage1-configs \
  --write-report runs/preflight/stage1-runs-4gpu.json
```

`prepare_stage1_runs.py` reads the four primary paragraph configs, checks that
per-device batch times world size divides global batch 512, and changes only
`gradient_accumulation_steps` in the rendered copies. It prints the exact
`torchrun` command for each copy and records the calculation in the optional
report. It does not touch the canonical configs, inspect the corpus, or launch
training. The two document follow-ups are outside this four-arm plan and are
launched from their configs after their local corpora pass validation.

The configs set `report_to` to `none`, so retain the cluster scheduler's
standard output and error log with the run artifacts.

At training time, `wtpsplit/train/stage1_data.py` first looks at the config's
`train_text_path` and `valid_text_path`. Existing local directories may contain
JSONL, JSON, Parquet, or CSV shards, with one format per directory. The mC4
paragraph configs fall back to the pinned `markus583/mC4-TEST` Hub revision
when their placeholder local paths are absent. FineWeb and document configs
have no Hub fallback and fail clearly when their local directories are missing.

Batch 64 still needs a short memory check on the target GPU, especially for
mmBERT. If it does not fit, lower the per-device batch and choose accumulation
so that their product with world size remains 512.

For a training smoke, copy one rendered config to a temporary location. Set
`max_steps` to 100, `save_steps` to 50, `logging_steps` to 10, and
`output_dir` to a smoke-only directory. Leave
`cleanup_checkpoints_after_training` false. Keep the full arm's data path,
tokenizer, punctuation file, precision mode, and distributed batch
calculation.

After the smoke finishes, reload the saved model and tokenizer through the SaT
backbone resolver. Replace the path below with the smoke output directory:

```bash
uv run python -c '
from transformers import AutoTokenizer
from wtpsplit.train.backbones import resolve_backbone
p = "runs/smoke/stage1_fineweb"
_, model_class, _ = resolve_backbone(p)
model_class.from_pretrained(p)
AutoTokenizer.from_pretrained(p)
print("checkpoint reload: ok")
'
```

The output directory must contain model weights, `config.json`, tokenizer
files, `training_args.bin`, and `trainer_state.json`. The reload command must
finish without missing or unexpected-weight errors. A smoke result is never an
experiment result.

Full runs keep the most recent `checkpoint-*` directory and save the final
reloadable model at `output_dir`. To resume an interrupted run, edit the
rendered config rather than the canonical config and set:

```json
"resume_from_checkpoint": "/scratch/mmsat/runs/curriculum/stage1_fineweb/checkpoint-150000"
```

Keep the original data receipt, seed, batch calculation, and model settings
when resuming. Start a new output directory if any of those inputs change.

## Evaluation and selection

Every final checkpoint is evaluated on the same UD, OPUS100, and Ersatz packet.
At present, BOUQuET and broader multilingual document evaluation are outside
the Stage 1 model-selection score. Adding another dataset requires a frozen
protocol and a new evaluation of every arm.

```bash
uv run python scripts/evaluate_stage1.py \
  --model runs/curriculum/stage1_fineweb \
  --output runs/curriculum/evaluation/stage1_fineweb.json
```

The wrapper creates an isolated cache and work directory, calls the historical
adaptation evaluator at threshold 0.01, captures its console output in
`evaluation.log`, and requires exactly one newly written intrinsic result. It
then writes `<name>.raw.json`, the compact `<name>.json`, and
`<name>.run.json` with command arguments and SHA-256 hashes. A failed evaluator
leaves the log path in the error message; ambiguous or missing result files are
treated as failures instead of selecting one silently.

The protocol and metric meanings are recorded in
[EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md). Compare language macro F1 at
the fixed threshold, tuned-threshold F1, weak languages and scripts,
throughput, failures, and checkpoint size. Apply the same comparisons to all
four paragraph arms. Treat the document runs as a separate unit check.

`configs/curriculum/selections.json` stays at `pending_matched_runs` until the
FineWeb and mmBERT cells have comparable final evaluations. Record the chosen
corpus, unit, backbone, checkpoint, and evidence paths there. Existing results
are collected in [STAGE1_RESULTS.md](STAGE1_RESULTS.md).

The handoff record for every completed arm consists of:

- the Git commit and canonical plus rendered config;
- the corpus `metadata.json` and final validation report, or the pinned mC4
  dataset revision for a Hub-backed arm;
- the run-plan report, training log, `trainer_state.json`, and final checkpoint;
- the compact evaluation, raw result, evaluation run receipt, and
  `evaluation.log`;
- any warnings accepted during corpus validation and the reason they were
  accepted.

Model weights, caches, and full logs belong in research storage. Compact
receipts, hashes, final metrics, and the selection decision belong in the
repository or the release record used for handoff.

## What still needs cluster access

The repository can resolve, validate, and dry-run the full plan now. Actual
completion requires the following work on a machine with source and GPU access:

- build and validate the matched FineWeb paragraph corpus;
- run checkpoint save-and-reload smokes for XLM-R and mmBERT;
- train and evaluate the three pending paragraph arms, and preserve the
  completed mC4/XLM-R result as the fourth comparison cell;
- build, train, and evaluate FineWeb documents if the unit follow-up remains
  useful;
- materialize the wider FineWeb corpus and add mixture-weighted sampling before
  attempting the 1,870-pair run.

No Stage 1 checkpoint should be promoted from a smoke or from the old
FineWeb-versus-raw Track B probe.
