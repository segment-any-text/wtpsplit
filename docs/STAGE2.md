# SaT Stage 2

**NOTE:** This is all very preliminary and just tests out some initial ideas.
Needs to be verified at scale; conclusions could also change depending on
implementation. Conclusions are merely (weak) suggestions!

**NOTE 2:** This note and others were mostly written up by agents to get an easier entrance into this project
and digest the early, very early results. Do not blindly trust them - double-check the
implementations, and especially the conclusions! This is all very preliminary;
take with a big grain of salt.

Stage 2 trains a sentence-boundary classifier from a Stage 1 larger-scale run. The
current handover covers a 21-language-script pilot, a matched replay experiment,
and the evaluation tools needed to replace the pilot metrics with exact scoring.

So: the code path is usable. The evidence is preliminary. Most completed runs used
one seed, short budgets, a local laptop GPU or a brief shared-cluster slot, and
small BOUQuET slices. See [STAGE2_RESULTS.md](STAGE2_RESULTS.md) before treating
any current selection as settled.

## Experiment tracks

The work split into several tracks. They answer different questions and should
remain separate in later runs.

| Track | Question | Current use |
| --- | --- | --- |
| Supervision | Are audited native, parallel, and forward-translated labels better than the historical sentence corpus? | Keep the A/B/C corpus; confirm under the exact evaluation contract. |
| Retention | Does historical replay preserve established-language performance while learning the 21-language pilot? | Next matched comparison: no replay versus 50% replay. |
| Backbone and head | XLM-R or mmBERT; token-final or character-resolution prediction? | XLM-R/token is the provisional main arm. A matched character-head config is included as an extra experiment; neither choice is settled. |
| Stage 1 transfer | Does web pretraining improve the same Stage 2 recipe? | Pending the full Stage 1 selection. The completed 20k-step probe was too short. |
| Stage 3 consolidation | Does historical corrupted sentence training retain new Stage 2 gains while restoring broad compatibility? | Optional follow-up after a Stage 2 checkpoint is selected. |
| Tier C versus Tier D supervision | Should synthetic labels come from forward sentence translation, or from reverse document translation with boundary projection? | Both remain open. Tier C is materialized in the current pilot; the tested Tier D variants were weak on one 25-document Dhivehi gate and need a broader, better-designed follow-up. |

The direct handover contains the materialized A/B/C path, retention,
token/character experiments, evaluation, and Stage 3. Tier D remains a parallel
research track rather than a discarded tier. Its present prototypes are not
part of the materialized corpus build; their findings and next questions are
recorded in the results page.

## How the stages connect

```text
Stage 1 checkpoint and tokenizer
              |
              v
UD train ------------------> Tier A --+
Tatoeba -------------------> Tier B --+--> A/B corpus --+
clean EWT -> sentence-wise NLLB -> Tier C ----------+--> A/B+C arm --+
                                                    |               |
native target docs -> document MT + projection -> Tier D            +--> selected supervision arm
                                                    |               |
A/B corpus -----------------------------------------+--> A/B+D arm --+
                                                                    |
historical SaT corpus --> replay control ---------------------------+--> Stage 2
                                                        |
                                                        +--> external evaluation
                                                        |
historical SaT corpus + corruption --------------------+--> optional Stage 3
```

Tier C forward sentence translation starts from trusted segmented English,
translates each sentence separately, and inherits the joins after
concatenation. Tier D reverse document translation with boundary projection
starts from a native target-language document, translates the whole document
to a pivot language, obtains boundary evidence there, and projects it back
onto the original target text. Tier D preserves native target documents and
could cover languages or domains for which the Tier C direction is unsuitable;
it also introduces translation, segmentation, and alignment uncertainty.

The current A/B/C artifact uses Tier C and one source per language. It is one
supervision arm, not a decision that Tier C has permanently replaced Tier D.
A future Tier-D corpus should be compared with the corresponding Tier-C arm
and with A/B alone before either synthetic route becomes a default.

Stage 1 and Stage 2 use the same backbone wrapper. Passing a Stage 1 directory
as both model and tokenizer loads its encoder weights and saved tokenizer,
including the added newline token. The checkpoint's model type selects the
wrapper. The Stage 2 configuration still sets the requested depth and
lookahead, and replaces the Stage 1 classifier with a one-label sentence
classifier.

Stage 3 starts from the selected Stage 2 output directory. It returns to the
historical 89-language sentence corpus and enables the historical ASR and
social-media corruption. Its purpose is consolidation. It can also erase
coverage-tail gains, so compare Stage 2 and Stage 2→3 rather than assuming the
extra stage helps.

## Pipeline map

| Gate | Command or file | What passes to the next gate |
| --- | --- | --- |
| Source audit | `audit_ud_tier_a.py`, `audit_tier_b_parallel.py` | accepted source rows, counts, licenses, contamination decisions, and the clean Tier-B corpus |
| A/B build | `build_stage2_ab_pilot.py` | capped Tier-A/B training corpus and manifest |
| Tier C build | `build_tier_c_forward.py` | sentence-wise NLLB translations and manifest |
| Merge | `build_stage2_abc_pilot.py` | one-source-per-language 21-pair corpus |
| Replay build | `build_historical_stage2_control.py` | 17-pair compatibility corpus with BOUQuET matches removed |
| Corpus gate | `validate_stage2.py` | config/schema checks, corpus inventory, and artifact hash verification |
| Run preparation | `prepare_stage2_runs.py` | matched no-replay and replay configs from one model initialization |
| Training | `train_SM.py` | loadable final model/tokenizer plus checkpoints and logs |
| Head comparison | `stage2.json`, `stage2_character.json` | matched token- and character-head outputs using the selected Stage 2 mixture |
| Evaluation data | `download_ud_plan.py`, `build_stage2_ud_evaluation.py`, `build_focused_evaluation_packet.py` | exact UD and partial BOUQuET JSONL packets |
| Model gate | `evaluate_mmsat.py` | thresholds fitted on exact dev rows and exact/partial test reports |
| Decision | `configs/curriculum/selections.json` | selected Stage 2 checkpoint and optional Stage 3 input |

The data sources, script inputs, outputs, and filtering rules are described in
[STAGE2_DATA.md](STAGE2_DATA.md).

## Run order

For a new machine, use this order:

1. Install the locked research environment and authenticate with Hugging Face.
2. Copy `data/all_data_11_05-all.pth`, or copy the final A/B/C and replay
   artifacts when rebuilding is unnecessary.
3. Run `validate_stage2.py` without `--require-data` to check the source-only
   handover.
4. Rebuild the source audits and corpora, or place copied artifacts at the paths
   recorded in their manifests.
5. Run `validate_stage2.py --require-data`. Resolve every hash or schema error.
6. Select the Stage 1 checkpoint. Use raw XLM-R only for the provisional
   backbone control.
7. Render the no-replay/replay pair and validate both rendered configs.
8. Run a short save-and-reload smoke for the replay arm.
9. Launch the matched pair. Preserve configs, logs, checkpoints, final model
   directories, and validation reports.
10. Build exact UD dev/test packets and the partial BOUQuET diagnostic packet.
11. Score both arms with thresholds fitted only on exact development rows.
12. Update `selections.json` with the retention decision.
13. Run the optional matched token/character comparison on that selected
    mixture. Run Stage 3 only as a matched follow-up to the selected Stage 2
    checkpoint.

## Setup and external artifacts

```bash
uv sync --locked --group research --extra legacy
hf auth login
uv run python scripts/validate_stage2.py
uv run pytest -q
```

The tests use miniature fixtures and synthetic corpora. They cover source
filtering, schema and hash validation, replay scheduling, run rendering, exact
versus partial evaluation, and both backbone wrappers. They do not rebuild the
upstream corpora, run NLLB, establish label quality, or substitute for a
training smoke on the target hardware.

A source-only checkout reports missing `.pth` files as warnings. The following
files stay out of Git:

| File | Required for |
| --- | --- |
| `data/all_data_11_05-all.pth` | rebuilding audits and replay; optional Stage 3 |
| `data/mmsat_stage2_abc_pilot_v1.pth` | primary Stage 2 training |
| `data/mmsat_stage2_historical_control_v1.pth` | matched replay arm |
| `data/mmsat_stage2_ab_pilot_v1.pth` | rebuilding the merge and Tier C source selection |
| `data/mmsat_stage2_tier_c_forward_v1.pth` | rebuilding the final merge |

The four derived Stage 2 corpus manifests record expected hashes. The separate
`sat_historical_corpus_v1.json` receipt covers the legacy source/Stage 3
corpus and records its provenance limits. A collaborator can either copy the
final A/B/C and replay files for training or rebuild the full chain from the
historical corpus and pinned upstream sources. Tier C reconstruction also
needs a GPU for practical runtime.

## Build and validate the data

The complete build is:

```bash
uv run python scripts/audit_ud_tier_a.py
uv run python scripts/audit_tier_b_parallel.py
uv run python scripts/build_stage2_ab_pilot.py

uv run python scripts/build_tier_c_forward.py --device cuda

uv run python scripts/build_stage2_abc_pilot.py
uv run python scripts/build_historical_stage2_control.py

uv run python scripts/validate_stage2.py --require-data \
  --write-report runs/preflight/stage2.json
```

The builders default to the BOUQuET and NLLB revisions recorded in the
manifests. Changing a revision creates a new corpus version and requires new
manifests and downstream results.

`validate_stage2.py` checks config keys, required paths, replay settings,
artifact hashes, supported dataset names, language counts, and the absence of
embedded evaluation data. It exits with status 2 on an invalid setup.

## Prepare the matched retention run

The checked-in configs are useful before Stage 1 has selected a checkpoint:

| Config | Input mixture | Purpose |
| --- | --- | --- |
| `stage2_no_replay.json` | A/B/C only | retention comparison control |
| `stage2.json` | A/B/C batches plus 50% historical replay batches | candidate retention treatment |
| `stage2_character.json` | same as `stage2.json`, with character-resolution output | optional matched head experiment |
| `stage3.json` | historical sentence corpus with corruption | optional post-Stage-2 consolidation |

Render the matched Stage 2 pair from the chosen Stage 1 output:

```bash
uv run python scripts/prepare_stage2_runs.py \
  --model runs/curriculum/SELECTED_STAGE1 \
  --tokenizer runs/curriculum/SELECTED_STAGE1 \
  --render-dir runs/stage2_retention/configs

uv run python scripts/validate_stage2.py \
  --config runs/stage2_retention/configs/no_replay.json --require-data

uv run python scripts/validate_stage2.py \
  --config runs/stage2_retention/configs/replay.json --require-data
```

Omit `--render-dir` for a dry plan. Existing rendered configs are not
overwritten unless `--overwrite` is passed.

Replay is scheduled at batch level. With `replay_fraction=0.5`, the sampler
alternates primary and replay batches in a deterministic weighted schedule.
Inside each corpus it retains the trainer's round-robin schedule across
language/dataset streams, cycling smaller streams as needed. The corpora are
loaded separately, so shared language identifiers cannot replace one another
in memory. The fraction describes emitted batches, not unique sentences,
tokens, languages, or corpus volume.

## Training and smoke checks

Launch the rendered configs directly:

```bash
uv run python wtpsplit/train/train_SM.py \
  runs/stage2_retention/configs/no_replay.json

uv run python wtpsplit/train/train_SM.py \
  runs/stage2_retention/configs/replay.json
```

For a smoke, render a separate pair with `--max-steps 1`, then set
`save_steps` to `1`, `save_total_limit` to `1`, and use a smoke-only output
root. Keep the real corpora and replay fraction. Reload the saved directory
before accepting the smoke.

`train_SM.py` writes the final model and tokenizer into `output_dir` after
training. Intermediate `checkpoint-*` directories remain resumable training
state; downstream evaluation and Stage 3 use the stable output directory.

The canonical Stage 2 configs disable embedded evaluation because these
training corpora intentionally contain no evaluation rows. Evaluation is a
separate gate.

## Optional character-head experiment

The default Stage 2 arm predicts at the final character of each subword token.
The optional character head predicts at every character position from the
covering token state, the token-level logit, hashed character features, and the
character's position inside the token. The implementation is in
`wtpsplit/char_head.py`; the XLM-R and mmBERT wrappers attach the same head in
`wtpsplit/models.py` and `wtpsplit/models_modernbert.py`. `train_SM.py` builds
the character labels and selects the character-aware collator.

`stage2_character.json` is matched to the checked-in replay arm. It differs
from `stage2.json` only in `output_dir`, `use_character_head`, and
`character_head_init`. After Stage 1 selection, render character versions of
the same no-replay/replay pair rather than editing checkpoint paths by hand:

```bash
uv run python scripts/prepare_stage2_runs.py \
  --model runs/curriculum/SELECTED_STAGE1 \
  --tokenizer runs/curriculum/SELECTED_STAGE1 \
  --head character --character-head-init random \
  --output-root runs/stage2_character \
  --render-dir runs/stage2_character/configs
```

This produces both mixture variants. Compare the character arm corresponding
to the retention winner with its token counterpart; do not compare a replay
character model with a no-replay token model. Keep backbone, tokenizer, corpus
mixture, optimizer, steps, seed, and evaluation packets fixed.

The checked-in character arm uses `character_head_init="random"` because it is
a from-the-same-encoder architecture comparison. `"identity"` is a different
experiment: it adds the head to a trained token-boundary checkpoint while
initially reproducing that checkpoint's token-final outputs. Such a warm-start
can be useful, but it is not the matched architecture comparison described
here.

Character tensors grow with the number of characters rather than the number of
subword tokens and depend on fast-tokenizer offset mappings. Record peak memory
and throughput alongside quality. The existing result is only 48 of 65 known
joins on a partial-reference packet, against 50 of 65 for the token head. Run
the exact UD evaluation, a larger coverage-tail set, and more than one seed
before selecting it. The present config makes the experiment reproducible; it
does not make the character head the Stage 2 default.

## Evaluation

The current tools support two reference types:

| Reference | Valid metrics | Limitation |
| --- | --- | --- |
| exact UD dev/test | precision, recall, F1; dev threshold fitting | treebank domains do not cover the full deployment setting |
| partial BOUQuET joins | known-boundary recall | unlabelled boundaries inside a BOUQuET segment prevent precision or F1 |

Download the frozen UD 2.18 files and build an exact packet:

```bash
uv run python scripts/download_ud_plan.py \
  --plan data/manifests/ud_2_18_frozen_selection_v1.json \
  --output data/external/ud-treebanks-v2.18-frozen \
  --selection all --split dev --split test \
  --max-download-mb 2048 --execute

uv run python scripts/build_stage2_ud_evaluation.py
```

Build and validate the small BOUQUET diagnostic packet:

```bash
uv run python scripts/build_focused_evaluation_packet.py --split test

uv run python scripts/validate_focused_evaluation.py \
  data/evaluation/focused_bouquet_test_candidates_v1.jsonl
```

That packet contains five documents for each of five languages. It is useful
for checking known boundaries and reported failure cases. It is far too small
for model selection, and it is not exact gold. The earlier Track B selection
used only 65 known boundaries from this packet; those results are provisional.

The BOUQuET segment audit explains the partial-reference restriction:

```bash
uv run python scripts/audit_bouquet_segments.py --split dev
```

On the current revision, two independent English detectors agree that at least
21 of 504 unique BOUQuET units contain multiple sentences. This is a lower
bound. Treating every unit as one complete sentence creates false apparent
over-splitting.

Score exact and partial packets together:

```bash
uv run python scripts/evaluate_mmsat.py \
  --input data/evaluation/stage2_ud_2_18_exact_v1.jsonl \
          data/evaluation/focused_bouquet_test_candidates_v1.jsonl \
  --model runs/stage2_retention/replay \
  --tokenizer runs/stage2_retention/replay \
  --output runs/stage2_retention/replay/evaluation.json \
  --scored-output runs/stage2_retention/replay/scored.jsonl
```

`evaluate_mmsat.py` fits thresholds on exact development rows only. Exact test
rows receive precision, recall, and F1. Partial rows receive known-boundary
recall; additional predictions are left unjudged.

Romansh and OCR still need source-disjoint, double-annotated development and
test sets. Validate such packets with `validate_focused_evaluation.py
--require-gold` before using them for final claims.

## Selection and optional Stage 3

Choose replay only after the exact report shows a useful tail-language gain
within a declared historical-language retention budget. Record the chosen
checkpoint and evidence path in `configs/curriculum/selections.json`.
That file also keeps the character-head comparison pending and records Tier D
reverse document translation with boundary projection as an open alternative
to Tier C, without pretending that a Tier-D training artifact already exists.

`stage3.json` expects the selected Stage 2 model at
`runs/curriculum/stage2`. Point that path to the chosen final output directory,
or render a private copy of the config with both `model_name_or_path` and
`tokenizer_name_or_path` set to the chosen directory. Then run:

```bash
uv run python wtpsplit/train/train_SM.py configs/curriculum/stage3.json
```

Evaluate the Stage 2 and Stage 2→3 outputs with the same packets, thresholds,
seeds, and retention slices. A Stage 3 run without that direct comparison does
not answer the consolidation question.

## What remains before scale-up

The 21-pair corpus tests the machinery and a small supervision panel. Scaling
to hundreds of language-script pairs needs a versioned source inventory,
licenses and training permissions, per-language quotas, contamination status,
translation coverage, build receipts, and mixture weights. Most of that
inventory is still research work outside this handover.

Run the exact replay comparison first. Then repeat the backbone/head comparison
on coverage-tail languages, run more than one seed for finalists, and decide
whether Stage 3 retains the gains. A large corpus build before those checks
would multiply unresolved choices.
