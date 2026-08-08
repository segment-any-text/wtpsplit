# Stage 1 data

Stage 1 has a controlled 85-language experiment and a later FineWeb expansion.
The controlled experiment answers the corpus and backbone questions. The
expansion aims to improve coverage across hundreds or thousands of
language-script pairs. They use different balancing rules and should produce
separate run families.

## Choose the data path first

The controlled experiment uses `scripts/build_stage1.py`. It streams pinned
web sources, filters FineWeb documents against the evaluation fingerprint
index, converts them to paragraphs or keeps them as documents, writes one
Parquet shard per language and split, and maintains `metadata.json`. This is
the path used by the curriculum configs in `configs/curriculum/`.

The large FineWeb expansion currently uses `scripts/sample_fineweb2.py` for
bounded acquisition and `scripts/prepare_stage1_filtered_splits.py` for pilot
training files. That route writes JSONL and separate receipts. It is useful for
access checks and small training probes. A full materializer that enforces the
frozen quotas and mixture weights across all 1,870 pairs is still pending.
`configs/mmsat_3l.json` is the experimental mmBERT pilot for this route. Its
batch size, evaluation behavior, and reporting settings are pilot choices, not
evidence for the matched four-arm comparison.

These paths should not be spliced together. A matched build already performs
contamination filtering and deterministic splitting. Running the sample-shard
filter after it would duplicate work and would discard its Parquet receipt.

## Script map

| Script | Reads | Writes | Role |
| --- | --- | --- | --- |
| `measure_mc4_test_char_mass.py` | downloaded `markus583/mC4-TEST` Parquet cache | per-language JSON and review CSV | Recomputes the character budgets used for the matched experiment. It is a provenance tool, not a normal prerequisite for every run. |
| `download_ud_plan.py` | frozen UD selection | selected `.conllu` files and `download_receipt.json` | Fetches the evaluation files needed to rebuild the contamination index; dry-run is the default. |
| `build_contamination_index.py` | frozen UD test files and BOUQuET dev/test | compressed fingerprint index and small manifest | Builds a text-free index used to exclude exact and conservative near evaluation duplicates. |
| `build_stage1.py` | character caps, source mapping, sampling plan, script remaps, fingerprint index | `train/*.parquet`, `valid/*.parquet`, `metadata.json` | Main resumable builder for matched mC4/FineWeb paragraphs and documents. |
| `validate_stage1.py` | one or more builder receipts | JSON report, optionally saved with `--write-report` | Checks corpus structure and provenance in memory proportional to the number of languages. |
| `build_fineweb2_coverage.py` | FineWeb2 distribution CSV and evaluation coverage | coverage JSON and review CSV | Refreshes the available language-script inventory for scale-out. |
| `build_stage1_sampling_plan.py` | coverage and evaluation manifests | quota plan and gap review, in JSON and CSV | Freezes document caps, validation routing, warnings, and normalized target weights. |
| `sample_fineweb2.py` | frozen sampling plan and remote FineWeb2 streams | bounded raw JSONL files and `download_receipt.json` | Tests access or acquires pilot samples; dry-run is the default. |
| `prepare_stage1_filtered_splits.py` | raw sample JSONL and fingerprint index | `.filtered.jsonl` train/valid trees and a split receipt | Filters and deterministically splits bounded scale-out samples for pilot training. |

`scripts/doctor.py`, `scripts/dry_run_stage1.py`,
`scripts/prepare_stage1_runs.py`, and `scripts/evaluate_stage1.py` cover the
machine, plan, launch, and model gates. Their order and exit behavior are in
[STAGE1.md](STAGE1.md).

## The matched corpus

The mC4 training split contains about 50.98 billion characters across the 85
historical SaT languages. Its per-language counts are the budgets for every
matched build. A source that runs out of usable text stays short; the builder
does not repeat it to meet the budget.

| Manifest | Purpose |
| --- | --- |
| `mc4_test_per_lang_char_mass.json` | mC4 revision and per-language character budgets |
| `sat_lang_to_fineweb2_v1.json` | SaT language to FineWeb source mapping |
| `stage1_fineweb_script_remaps_v1.json` | corrections for misleading Latin-script mappings |
| `mmsat_contamination_index_v1.json` | checksum and provenance for the external evaluation fingerprint index |

The main mC4 paragraph arm reads the pinned `markus583/mC4-TEST` snapshot.
Rebuilding mC4 locally is available for symmetric checks, although the live
`allenai/c4` stream is not a bitwise copy of that snapshot.

FineWeb uses FineWeb2 for most languages and the original FineWeb dataset for
English. Evaluation contamination is checked at document level before a
document is split into paragraphs. This order matters because one overlapping
document could otherwise create several training rows.

### Recomputing the mC4 budgets

The checked-in character manifest is frozen, so ordinary builds consume it
directly. To reproduce it, first make the pinned `markus583/mC4-TEST` Parquet
snapshot available in the Hugging Face cache, then run:

```bash
HF_HOME=/scratch/huggingface hf download markus583/mC4-TEST \
  --repo-type dataset \
  --revision 6c109b67925b989746262bb67f0214f59bb1f8a2

uv run python scripts/measure_mc4_test_char_mass.py \
  --hf-home /scratch/huggingface \
  --splits train valid \
  --output data/manifests/mc4_test_per_lang_char_mass.json
```

The scan reads `lang`, `text`, and `ends_with_punctuation` from every cached
Parquet shard. The JSON contains row and character counts for each language;
the sibling CSV is a review surface. Recomputing the file is a manifest change
and should be reviewed before any matched run uses the new caps. The scanner
selects only the requested Hub snapshot, and writes that revision into the
manifest, so another cached snapshot cannot be used by accident.

### Building the contamination index

FineWeb materialization requires the compressed index at
`data/external/mmsat_contamination_index_v1.json.gz`. For the matched
experiment, prefer copying the existing artifact and checking it against
`data/manifests/mmsat_contamination_index_v1.json`.

The current manifest predates complete retention of the BOUQuET source
revision, so the original index cannot be rebuilt byte for byte from the
repository alone. Rebuilding deliberately creates a refreshed contamination
surface. Pin the BOUQuET revision, review the new manifest, and rebuild the
FineWeb corpus that consumes it:

```bash
uv run python scripts/download_ud_plan.py \
  --plan data/manifests/ud_2_18_frozen_selection_v1.json \
  --output data/external/ud-treebanks-v2.18-frozen \
  --selection all \
  --split evaluation \
  --no-metadata \
  --max-download-mb 1000

uv run python scripts/download_ud_plan.py \
  --plan data/manifests/ud_2_18_frozen_selection_v1.json \
  --output data/external/ud-treebanks-v2.18-frozen \
  --selection all \
  --split evaluation \
  --no-metadata \
  --max-download-mb 1000 \
  --execute

uv run python scripts/build_contamination_index.py \
  --ud-root data/external/ud-treebanks-v2.18-frozen \
  --ud-selection data/manifests/ud_2_18_frozen_selection_v1.json \
  --bouquet-revision BOUQUET_COMMIT_OR_TAG \
  --output data/external/mmsat_contamination_index_v1.json.gz \
  --manifest data/manifests/mmsat_contamination_index_v1.json
```

The downloader previews every request until `--execute` is supplied. It uses
atomic `.part` files, respects the cumulative byte budget, reuses cached files,
and writes sizes and SHA-256 hashes to `download_receipt.json`. Any failed
request makes it exit 1.

The index builder then verifies each selected UD file against its frozen
SHA-256 before reading sentence text. It also loads BOUQuET dev and test unless
`--skip-bouquet` is supplied. The compressed artifact stores normalized exact
and near-match fingerprints rather than the evaluation text. Git tracks its
small manifest and checksum; the index itself stays in external storage.

The build fails on a missing UD file or checksum mismatch. Supplying
`--skip-bouquet` changes the contamination surface and should only be used for
an explicitly labelled diagnostic corpus.

## Matched local builds

Start with source resolution. This checks mappings, revisions, selected
languages, and caps without opening the remote datasets:

```bash
uv run python scripts/build_stage1.py \
  --corpus fineweb2 \
  --unit paragraph \
  --dry-run
```

The builder pins FineWeb2, FineWeb English, and C4 revisions. FineWeb uses the
sampling plan and script-remap manifest to resolve sources. mC4 uses the SaT
language mapping and C4 source configurations when a local rebuild is
requested.

Build the matched FineWeb paragraphs with:

```bash
uv run python scripts/build_stage1.py \
  --corpus fineweb2 \
  --unit paragraph \
  --output-dir data/external/fineweb2-stage1-paragraphs
```

Document builds use the same budgets:

```bash
uv run python scripts/build_stage1.py \
  --corpus mc4 \
  --unit document \
  --output-dir data/external/mc4-stage1-documents-matched

uv run python scripts/build_stage1.py \
  --corpus fineweb2 \
  --unit document \
  --output-dir data/external/fineweb2-stage1-documents-matched
```

Each build writes one shard per language under `train/` and `valid/`, plus a
single `metadata.json`. The receipt records source revisions, budgets, counts,
artifact hashes, contamination statistics, and the build fingerprint. A
completed language is reused only when its recorded artifacts still match.

Writes are atomic at shard level: temporary `.partial` files are renamed after
a language succeeds. On restart, completed languages with matching hashes are
reused. A changed source revision, cap file, mapping, unit, split ratio, seed,
or contamination input changes the build fingerprint and prevents accidental
resumption into an incompatible directory.

For FineWeb, contamination filtering is enabled by default and the index must
exist. `--skip-contamination-filter` is reserved for pipeline diagnostics; its
receipt will fail validation when `--require-contamination-filter` is used.
The mC4 Hub paragraph arm preserves the historical training source and is
therefore handled separately from a newly filtered FineWeb build.

A small build can exercise every selected language without changing the frozen
budgets:

```bash
uv run python scripts/build_stage1.py \
  --corpus fineweb2 \
  --unit paragraph \
  --max-chars-per-language 10000 \
  --dry-run
```

Remove `--dry-run` to materialize it. `--cap-scale 0.001` provides a
proportional smoke instead of a fixed ceiling. Smoke output belongs in its own
directory because its fingerprint and budgets differ from the full corpus.

## Validation

`scripts/validate_stage1.py` checks the receipt without loading the whole
corpus into memory. It verifies language coverage, completion, source
revisions, unit type, required columns, train and validation counts, artifact
existence, split overlap, cap fill, and contamination state. Error output is
bounded for large language sets.

```bash
uv run python scripts/validate_stage1.py \
  data/external/fineweb2-stage1-paragraphs/metadata.json \
  --mapping data/manifests/sat_lang_to_fineweb2_v1.json \
  --require-contamination-filter \
  --verify-hashes
```

Hash verification reads every artifact and is best reserved for the final
corpus, a copied corpus, or an unexplained validation failure.

The structural pass checks all expected language entries, completion state,
unit, build fingerprint, pinned source revisions, required columns, non-empty
train and validation splits, split overlap, artifact existence and sizes, row
counts, cap fill, and contamination status. `--minimum-cap-fill` controls when
a short source is warned about. `--allow-incomplete` permits unfinished
languages during monitoring. `--max-reported-issues` bounds printed examples
while retaining the full error and warning counts, which keeps reports usable
for thousands of languages.

Several receipts can be checked in one invocation:

```bash
uv run python scripts/validate_stage1.py \
  data/external/fineweb2-stage1-paragraphs/metadata.json \
  data/external/fineweb2-stage1-documents-matched/metadata.json \
  --mapping data/manifests/sat_lang_to_fineweb2_v1.json \
  --require-contamination-filter \
  --write-report runs/preflight/fineweb-corpora.json
```

Exit code 0 means every supplied corpus is valid; exit code 2 means at least
one failed. Warnings such as a source exhausting below its character cap do
not change the exit code. Inspect both the boolean result and warning counts
before launching a matched comparison.

## Source mappings and shortfalls

Several FineWeb2 language codes initially resolved to small Latin-script
collections even when the SaT language uses another script. The tracked remap
manifest selects Arabic, Bengali, Devanagari, Gujarati, Kannada, Malayalam,
Tamil, Telugu, and other native-script configurations where appropriate. New
builds apply those mappings from the start.

Cebuano has no alternate script source and may finish below its mC4 budget.
Shortfalls are recorded in `metadata.json`; they are warnings unless the source
is empty or the experiment requires exact mass matching. The mC4 document
build reached 50.52 billion of 50.98 billion requested characters, with eight
languages exhausting their source.

Older notes describe top-up scripts that repaired an already running build.
Fresh builds should use the remap manifest and resumable builder instead.

## FineWeb scale-out

The current inventory contains 1,870 streamable language-script pairs and a
target of 6,601,467 documents. The plan includes 1,123 source-composition
warnings, often for Bible-heavy collections. Those warnings require review;
they do not automatically remove the source.

| Manifest | Purpose |
| --- | --- |
| `mmsat_dataset_coverage_v1.json` | evaluation-facing language-script universe |
| `fineweb2_stage1_coverage_v1.json` | available FineWeb2 sources and resource flags |
| `fineweb2_stage1_sampling_plan_v1.json` | document caps, validation routing, and target mixture weights |
| `stage1_coverage_gap_review_v1.json` | missing or mismatched evaluation identities |

The scale-out plan keeps micro sources, caps larger sources by priority, and
reduces large Bible-heavy sources to 5,000 documents. It does not apply the
matched experiment's equal-character budgets across all 1,870 pairs.

Preview bounded acquisition with:

```bash
uv run python scripts/sample_fineweb2.py
```

One source can be downloaded after access is available:

```bash
uv run python scripts/sample_fineweb2.py \
  --language-script abq_Cyrl \
  --max-documents-per-language 1000 \
  --max-download-mb 250 \
  --execute
```

The inventory, quotas, bounded sampler, and contamination policy are ready.
The full scale-out still needs a unified paragraph/document materialization
receipt and a training sampler that consumes `target_mixture_weight`. Until
those pieces exist, its status is `planned_not_materialized`.

### Refreshing the scale-out plan

The inventory refresh can fetch the official FineWeb2 distribution CSV, or
read a saved copy for an offline and reproducible rebuild:

```bash
uv run python scripts/build_fineweb2_coverage.py \
  --distribution-csv /path/to/fineweb2-language-distribution.csv

uv run python scripts/build_stage1_sampling_plan.py
```

The first command joins the FineWeb2 distribution with the evaluation coverage
manifest and records resource buckets, official test availability, source
composition flags, and acquisition metadata. The second applies priority caps,
retains micro sources, assigns official-test or deterministic validation, and
normalizes `target_mixture_weight`. It also writes a gap review without silently
substituting another script or language variety.

The JSON files are the machine-readable inputs. The sibling CSV files are
review surfaces and need not be committed when they contain the same rows.
Review changes in pair count, total target documents, coverage gaps, and source
composition warnings before replacing a frozen plan.

### Fetching and filtering a pilot

`sample_fineweb2.py` previews by default. Execution is bounded by language
count, documents per language, and total bytes:

```bash
uv run python scripts/sample_fineweb2.py \
  --language-script abq_Cyrl \
  --language-script ace_Latn \
  --max-documents-per-language 1000 \
  --max-download-mb 250 \
  --execute
```

The command streams remote rows, uses a seeded shuffle buffer, writes each
language atomically, reuses a non-empty cached shard, and records document
counts, bytes, output hashes, the sampling-plan hash, and each pinned dataset
revision in `download_receipt.json`. It refuses more than 25 selected pairs
during execution unless `--allow-many` is present. Failures leave no partial
target file and make the command exit 1.

Raw samples have not passed the contamination gate. Filter and split them with:

```bash
uv run python scripts/prepare_stage1_filtered_splits.py \
  --input-dir data/external/fineweb2-stage1-samples \
  --train-dir data/external/fineweb2-stage1-samples/train \
  --valid-dir data/external/fineweb2-stage1-samples/valid \
  --index data/external/mmsat_contamination_index_v1.json.gz \
  --receipt data/manifests/stage1_filtered_splits_v1.json
```

The splitter removes exact matches and excludes conservative near matches. It
assigns the remaining rows by a stable hash of seed and document ID, producing
`.filtered.jsonl` files that `configs/mmsat_3l.json` requires. Invalid rows are
counted and excluded. Its receipt hashes the contamination index and both
output splits for every language. Existing outputs require an explicit
`--overwrite`.

The matched-corpus validator consumes `build_stage1.py` metadata and can also
check a future scale-out receipt with `--scaleout-plan`. The current bounded
JSONL sampler does not yet emit that receipt schema, so its download and split
receipts must be inspected separately. This missing unified materializer and
validator is one reason the 1,870-pair track remains a plan rather than a
train-ready corpus.

Large corpora, the compressed contamination index, and Hugging Face caches
stay outside Git. Configs, manifests, checksums, and compact validation reports
belong in the repository.
