# Stage 2 pilot results

This work was done quickly with limited local compute and a short shared-cluster
window. Most runs use seed 42 only. The completed experiments are useful for
diagnosing weak variants and choosing the next controlled run. They do not
support paper-level model comparisons or a scale-up decision.

## Evidence boundary

Several evaluation setups appear in the local results. Their numbers are not
interchangeable.

| Setup | Scope | What it can support | Main limitation |
| --- | --- | --- | --- |
| Local A/B/C pilot | 19 targets plus English/German; 300 steps; up to 120 BOUQuET dev and test documents and 400 known joins per language | relative comparison under one short recipe | BOUQuET segment interiors are incomplete, so the recorded precision and F1 are proxy values rather than exact-boundary metrics |
| Track B focused packet | 25 documents across Romansh, Igbo, Swati, Thai, and Khmer; 65 known joins total | a quick known-boundary recall check | tiny sample, partial references, fixed threshold 0.25, and ±3-character matching; small score differences can be a few boundaries |
| Historical Stage 3 2×2 | 659 held-out datasets across 89 historical language entries; 1,000-step arms; one seed | compatibility check on the old SaT population | poor coverage of the scripts and tokenizer failures that motivate mmBERT and character-resolution work |
| Tier D reverse-document projection gate | 25 Dhivehi development documents | diagnosis of weaknesses in the tested translation/alignment variants | one language, one small development sample, and no evidence that the Tier D direction itself fails more broadly |

The exact UD packet builder and frozen evaluator were added after these pilots.
No completed result in this page has yet gone through the full exact-dev,
source-disjoint test, multi-seed protocol. The clean-machine one-step smoke
checks the software path only.

The BOUQuET audit reinforces this restriction. Among 504 unique English
development units, PySBD and SaT agree that 21 contain multiple sentences and
disagree on another 26. The 21 are a lower bound. Predictions inside those
units were counted as false positives by the older proxy F1 runs even though
the reference does not establish that.

## Tracks and current decisions

| Track | Observation | Working decision |
| --- | --- | --- |
| A/B/C supervision | Under the 300-step local proxy, A/B/C exceeds the historical compatibility corpus by 0.0670 target macro F1 and 0.0050 on English/German. | Keep A/B/C fixed for the next comparison. Confirm with exact scoring. |
| Retention | Full adaptation gains 0.0064 on targets and loses 0.0177 on English/German relative to its initialization. | Compare no replay with 50% representative replay before scaling. |
| Tier C | On the 65-boundary Track B packet, ABC known-boundary recall is 0.677 and AB is 0.631. | Keep sentence-wise forward translation in the pilot. The difference is three recovered boundaries. |
| Backbone and head | On the same packet, XLM-R/token recovers 50/65 joins; XLM-R/character and mmBERT/token recover 48/65; mmBERT/character recovers 40/65. | Keep XLM-R/token as the provisional main arm and retain the checked-in character config as an extra matched experiment. This packet is too small to settle the architecture. |
| Stage 1 transfer | A 20k-step FineWeb Stage 1 probe followed by Stage 2 recovers 44/65 joins, compared with 50/65 for raw Stage 2. | Inconclusive. The intended Stage 1 budget is 200k and only one small metric was used. |
| Stage 3 consolidation | XLM-R/token leads the historical 89-language 2×2; the character arms are worse. | Treat this as compatibility evidence. Repeat on coverage-tail exact data before choosing a backbone or head for mmSaT. |
| Tier D reverse document translation with boundary projection | Clean paired English reaches 0.577 boundary F1; realistic whole-document MT falls to 0.299. Later variants also remain weak on the same narrow sample. | Keep Tier D open as an alternative to Tier C. Do not mix the current labels into A/B/C; design a broader matched Tier C/Tier D comparison. |
| Segmental CRF | Tested variants lose to the local pointwise and constrained-decoding baselines. | Do not use it as the default decoder. |

These are run-planning decisions. The replay result, exact evaluation, stronger
baseline, and more than one seed can change them.

## A/B/C supervision pilot

The selected artifact has 11 Tier-A, four Tier-B, and six Tier-C
language-script pairs: 98,445 training sentences with empty embedded evaluation
lists. Its SHA-256 is in
`data/manifests/mmsat_stage2_abc_pilot_v1.json`.

The local equal-step comparison kept the initialization, optimizer, 300-step
budget, batch size, seed, context fitting, and 50% English/German replay fixed.
It changed the supervision corpus:

| Slice | Historical | A/B/C | Difference |
| --- | ---: | ---: | ---: |
| 19 target languages | 0.6286 | 0.6955 | +0.0670 |
| English/German controls | 0.8367 | 0.8417 | +0.0050 |
| Tier A | 0.7357 | 0.7652 | +0.0295 |
| Tier B | 0.6416 | 0.7800 | +0.1385 |
| Tier C | 0.4591 | 0.5347 | +0.0755 |

These are the older BOUQuET proxy F1 values described above. They suggest that
the combined source-routing policy is worth retaining. They do not measure the
individual contribution of each source, and the incomplete references prevent
an exact interpretation of precision or over-splitting.

The comparison also favours A/B/C partly because it covers four targets absent
from the historical control: Egyptian Arabic, Tibetan, Dzongkha, and Swati.
That is part of the intended data-policy treatment, though it prevents a claim
that cleaning alone caused the difference.

## Retention pilot

Relative to the unadapted initialization, the same A/B/C run changes target
macro F1 from 0.6891 to 0.6955 and English/German from 0.8594 to 0.8417. Ten of
19 targets improve and nine regress. A head-only update reduces target macro F1
to 0.6592 and control macro F1 to 0.8497.

This is the reason for the checked-in replay/no-replay pair. The local runs did
not execute that pair under the new exact evaluator. There is no completed
result yet for the current 50% historical replay implementation.

## Track B architecture and Stage 1 probes

Track B used five BOUQuET candidate documents per language for five languages.
The scorer counted 65 known joins and allowed a prediction within three
characters of each join. At that scale:

- 0.769 means 50 of 65 joins recovered;
- 0.738 means 48 of 65;
- 0.677 means 44 of 65;
- 0.631 means 41 of 65.

The architecture gap between XLM-R/token and mmBERT/token is therefore two
known boundaries on a partial-reference packet. The data-policy gap between
ABC and AB is three. These results are adequate for ordering a cheap follow-up,
and too small for an architecture or corpus claim.

The checked-in `stage2_character.json` makes that follow-up explicit. It is not
a retrospective claim that the character head worked: no character-head run
has yet completed the exact Stage 2 evaluation contract, and the extra
character tensors have not been profiled at the intended training scale.

The Stage 1 transfer probe also used 20,000 steps instead of the planned
200,000. Its lower Track B score does not establish that Stage 1 pretraining is
unhelpful.

## Historical Stage 3 2×2

The historical comparison covers 659 held-out datasets and 89 language entries,
so it is broader than Track B. XLM-R/token has global F1 0.5807, followed by
mmBERT/token at 0.4950, XLM-R/character at 0.4886, and mmBERT/character at
0.3432.

The population is inherited from the old SaT corpus and has little XLM-R
unknown-token pressure. It answers a backward-compatibility question. It gives
weak evidence about Tibetan, Ol Chiki, Cherokee, Thaana, and the other coverage
tail that motivated the backbone and character-head work. The arms also used
1,000 steps and one seed.

Small character-supervision probes on Tibetan, Dzongkha, and Santali show that
targeted labels can improve those targets while full updates damage controls.
A 50:50 target/control replay variant recovers much of the control performance.
Those probes justify testing replay; they do not establish a final character
recipe.

## Tier D: reverse document translation with boundary projection

Tier C and Tier D are alternative synthetic-supervision routes. Forward Tier C
translates trusted English sentences separately and inherits the joins between
translated units. This gives clean join labels but trains on text shaped by
sentence-wise translation.

Tier D starts with a native target-language document, translates the whole
document to a pivot language, obtains boundary evidence there, and projects
that evidence back to positions in the original document. Its attraction is
the native target text and the possibility of covering languages or domains
where the Tier C translation direction is unavailable or undesirable.

The 25-document Dhivehi gate shows a large drop between clean paired English
and realistic whole-document MT. The attempted hard alignment, MT-attention,
backtranslation, learned-selection, translation-likelihood, and anchor
variants did not recover enough reliable labels in that sample. This locates
problems in the current translation, boundary-count, and projection designs.
It is not a broad comparison of Tier C and Tier D.

Tier D therefore remains open, but without an accepted training artifact yet.
The next useful experiment is a multilingual, exact-boundary comparison of A/B
alone, A/B plus Tier C, and A/B plus Tier D, with the synthetic routes matched
for languages, source domains, retained documents, model, and training budget.
Projection coverage and confidence filtering must be reported alongside
boundary quality: a high score on a tiny retained fraction would not establish
a useful corpus route.

## Evidence locations

Machine-readable pilot results remain under `data/diagnostics/` on the research
machine and are ignored by Git. The main files are:

- `mmsat_stage2_abc_pilot_cuda.json`
- `mmsat_stage2_historical_equal_step_control_cuda.json`
- `mmsat_stage2_abc_head_only_pilot_cuda.json`
- `mmsat_tier_c_clean_forward_pilot_cuda.json`
- `stage3_2x2_cuda.json`
- `track_b_rk10/architecture_decision.json`
- `track_b_rk10/data_policy_window_kbr.json`
- `track_b_rk10/stage1_value_window_kbr.json`

The handover can run without these local result files. The summary preserves
their scope and working decisions. Copy the raw diagnostics separately when a
collaborator needs to audit or reanalyse the pilots.
