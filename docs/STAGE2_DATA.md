# Stage 2 data

The current A/B/C build uses one sentence-label source per language-script
pair: native UD where available, then human sentence-aligned Tatoeba, then
Tier C forward sentence translation. It contains 21 pairs and 98,445
sentences. Tier D is an alternative synthetic arm, not simply a fourth
fallback appended to this routing order.

The current materialized pilot uses Tier C forward sentence translation.
Tier D reverse document translation with boundary projection is an open
alternative: it starts from native target documents, translates the document
to a pivot language, segments or scores boundaries there, and projects that
evidence back to target offsets. The tested variants were weak on a small
Dhivehi development sample, so Tier D is not mixed into this A/B/C artifact.
That local result narrows the next experiment; it does not close the route.

## Data flow

```text
pinned UD 2.18 ------------------------> Tier A --+
pinned Tatoeba + historical eval hashes -> Tier B --+--> A/B
A/B English EWT + pinned NLLB ----------> Tier C --------> A/B/C
historical SaT corpus + A/B/C language list ----------------> replay control

Alternative synthetic route, not materialized here:
native target documents -> document MT -> pivot boundaries -> projection -> Tier D
```

Every builder writes a `.pth` corpus and/or a small JSON manifest. The corpora
stay out of Git. The manifests preserve counts, source decisions, revisions,
and SHA-256 hashes.

## Script map

| Script | Reads | Writes | Why it exists |
| --- | --- | --- | --- |
| `audit_ud_tier_a.py` | pinned UD 2.18 URLs and historical SaT evaluation rows | Tier-A JSON/CSV audit | checks train/dev/test availability, licenses, duplicates, and exact evaluation overlap before native data is accepted |
| `audit_tier_b_parallel.py` | historical SaT corpus, pinned Tatoeba archives, pinned BOUQuET dev/test | Tier-B JSON/CSV audit and clean Tatoeba `.pth` | replaces opaque historical OPUS routing with auditable sentence-aligned sources where volume and license permit |
| `build_stage2_ab_pilot.py` | Tier-A and Tier-B audits plus the clean Tier-B `.pth` | capped 15-pair A/B corpus and manifest | materializes accepted sources, removes designated evaluation matches, and applies deterministic per-language caps |
| `build_tier_c_forward.py` | clean English EWT sentences from A/B, pinned NLLB, pinned BOUQuET | six-pair Tier-C corpus and manifest | translates each English sentence separately so target joins are inherited without word alignment |
| `build_stage2_abc_pilot.py` | A/B and Tier-C corpora | merged 21-pair corpus and manifest | enforces one source per language, supported dataset names, non-empty text, and no embedded evaluation rows |
| `build_historical_stage2_control.py` | historical SaT corpus, A/B/C language list, pinned BOUQuET | 17-pair replay/control corpus and manifest | provides an equal-format compatibility stream while excluding exact BOUQuET matches |
| `validate_stage2.py` | training config, manifests, local corpora | JSON validation report | catches missing files, hash mismatches, unknown config keys, invalid replay settings, and corpus-schema errors before training |

The CSV audits are convenient review exports. The JSON manifests are the
checked-in records consumed by the pipeline.

## Source policy

| Tier | Accepted source | Boundary label | Current scope |
| --- | --- | --- | ---: |
| A | native UD train split | upstream sentence unit | 11 pairs, 76,565 materialized sentences |
| B | human sentence-aligned Tatoeba | join between target sentence units | 4 pairs, 19,000 materialized sentences |
| C | NLLB translation of separate English EWT sentences | inherited join between translated units | 6 pairs, 2,880 retained sentences |
| D | reverse document translation with boundary projection | boundary evidence projected onto the original native document | open research arm; no accepted training artifact yet |

The A/B/C merge rejects a language found in more than one input. This keeps the
pilot interpretable and prevents loader priority from silently deciding which
source wins.

Tier A audited 13 candidates. Eleven were accepted. Ukrainian and Hindi UD
sources were held out of Tier A because their release-specific licensing needs
review; both have accepted Tier-B routes for the internal pilot.

Tier B audited six Tatoeba routes. Ukrainian, Khmer, Hindi, and Georgian met
the 500-sentence pilot floor. Igbo and Amharic had 23 and 205 clean sentences,
so they moved to Tier C.

Tier C uses the same deterministic 500-sentence English EWT source sample for
six targets. Empty, English-identical, duplicate, and exact BOUQuET-overlap
translations are removed. The retained count is 2,880. This is a coverage
probe, not enough data to establish a scalable synthetic-data recipe.

Tier C and Tier D solve the synthetic-supervision problem in opposite
directions. Tier C has clean inherited joins but produces translation-shaped
target text. Tier D retains naturally occurring target text but must infer
where boundaries belong after document translation and alignment. Later
experiments should compare both routes on the same languages, source domains,
training budget, and exact target-language evaluation. A combined corpus is a
third arm and should not substitute for that comparison.

## Filtering boundary

Training builders remove exact normalized matches to their designated
evaluation material:

- Tier A excludes the selected UD dev/test rows and BOUQuET dev/test rows;
- Tier B excludes historical embedded evaluation rows and BOUQuET dev/test;
- Tier C excludes BOUQuET dev/test after translation;
- the historical replay control excludes BOUQuET dev/test.

This is exact-match protection. It does not detect paraphrases, translations,
shared documents with different normalization, or upstream contamination that
cannot be reconstructed from the historical corpus. The historical corpus
lacks recoverable per-example OPUS provenance and is a compatibility control,
not release training data.

## Build from the historical corpus

Place `data/all_data_11_05-all.pth` at the documented path, authenticate with
Hugging Face, and run:

```bash
uv run python scripts/audit_ud_tier_a.py
uv run python scripts/audit_tier_b_parallel.py
uv run python scripts/build_stage2_ab_pilot.py

uv run python scripts/build_tier_c_forward.py --device cuda

uv run python scripts/build_stage2_abc_pilot.py
uv run python scripts/build_historical_stage2_control.py
```

Pinned inputs:

| Input | Revision |
| --- | --- |
| UD | release `r2.18` in the Tier-A source table |
| Tatoeba | OPUS release `v2026-07-08` |
| BOUQuET | `9a6070a9652e350dda1d353c4fd198533199a911` |
| NLLB | `f8d333a098d19b4fd9a8b18f94170487ad3f821d` |

The Tier-C defaults reproduce the checked artifact: 500 English source
sentences, batch size 16, greedy model generation, and the six languages listed
in its manifest. A revision, language list, sentence cap, or filtering change
creates a new artifact version.

## Artifacts

| Corpus | Languages | Sentences | Role |
| --- | ---: | ---: | --- |
| `mmsat_stage2_ab_pilot_v1.pth` | 15 | 95,565 | native and human-aligned supervision |
| `mmsat_stage2_tier_c_forward_v1.pth` | 6 | 2,880 | forward-synthetic complement |
| `mmsat_stage2_abc_pilot_v1.pth` | 21 | 98,445 | selected primary Stage 2 input |
| `mmsat_stage2_historical_control_v1.pth` | 17 | 124,839 | compatibility baseline and replay stream |

`data/manifests/sat_historical_corpus_v1.json` is the receipt for the original
`all_data_11_05-all.pth` source and optional Stage 3 corpus. Its hash establishes
which historical artifact is in use. It cannot restore the missing per-example
OPUS provenance or turn the embedded evaluations into an external test set.

The historical control covers 17 of the 21 A/B/C pairs. Egyptian Arabic,
Tibetan, Dzongkha, and Swati have no mapped historical source. This imbalance
matters when interpreting replay: a 50% replay rate does not give every target
language equal historical support.

Validate copied or rebuilt artifacts before rendering runs:

```bash
uv run python scripts/validate_stage2.py --require-data \
  --write-report runs/preflight/stage2.json
```

The validator compares bytes against the checked manifests and opens each
corpus through the same schema adapter used by training. A successful report
establishes artifact identity and loadability. It says nothing about label
quality beyond the recorded audits.

## Evaluation data stays separate

Stage 2 training `.pth` files contain empty evaluation lists. Exact UD and
partial BOUQuET packets are built as JSONL after training. This prevents the
trainer's historical embedded-evaluation path from becoming an accidental
second protocol.

The token- and character-head arms use the same `.pth` corpus. Sentence joins
remain the supervision source; `train_SM.py` expands them into character
positions at packing time when `use_character_head=true`. No separate
character-labelled corpus or manifest is required. This keeps the head
comparison about output resolution rather than a change in training data.

`build_stage2_ud_evaluation.py` groups pinned UD sentences into four-sentence
documents and records exact character offsets. `build_focused_evaluation_packet.py`
groups BOUQuET segments for five reported failure/deployment languages. Those
joins are known, while boundaries inside a segment may be missing.

The BOUQuET audit found 21 consensus multi-sentence units among 504 unique
English units on development, plus 26 detector disagreements. The 21 cases are
a lower bound. BOUQuET candidate packets support known-boundary recall and
cannot support precision or F1 without manual interior annotation.

## Scaling the data path

The current builders use explicit small source tables. A larger build needs one
versioned row per language-script pair with:

- chosen tier and fallback tier;
- source URL, revision, split, license, and training permission;
- document, sentence, boundary, and character counts;
- contamination inputs and filter result;
- translator support and retained fraction for Tier C;
- native-document source, pivot translator, projector, confidence rule, and
  retained fraction for Tier D;
- materialization state, artifact path, receipt, and hash;
- training weight and evaluation availability.

The source decision and the materializer should remain separate. A failed or
unlicensed source can then be rerouted without changing training code. The
current 21-pair manifests are pilot records; they are not a hundreds-language
inventory.

Character-head scale tests also need character-count distributions, tokenizer
offset checks, and memory/throughput measurements by script. Token counts alone
do not predict the size of the character tensors, especially for scripts that
the selected tokenizer fragments poorly.
