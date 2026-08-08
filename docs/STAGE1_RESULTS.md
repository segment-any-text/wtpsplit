# Stage 1 results

Two matched mC4 XLM-R runs have finished: paragraphs and documents. FineWeb
and mmBERT have not yet produced comparable final checkpoints. The current
evidence settles the unit choice for mC4, while the corpus and backbone
questions remain open.

## Completed runs

Both full runs used 85 languages, 200,000 steps, global batch 512, seed 42,
XLM-R trimmed to three layers, lookahead 48, and the same per-language
character budgets.

| Metric | mC4 paragraphs | mC4 documents |
| --- | ---: | ---: |
| Language macro F1 at fixed threshold | 0.8324 | 0.830 |
| Language macro F1 at tuned threshold | 0.8593 | 0.858 |
| UD F1 at fixed threshold | 0.9127 | 0.911 |
| OPUS100 F1 at fixed threshold | 0.7807 | 0.778 |
| Ersatz F1 at fixed threshold | 0.9337 | 0.932 |
| English three-corpus macro | 0.9260 | 0.935 |

The difference in language macro F1 is about 0.2 percentage points. Spearman
correlation between per-language means is 0.994. Forty-seven of 85 languages
fall within half a percentage point, and only one language moves by more than
five points at the fixed threshold. Georgian is that outlier; its gap largely
disappears after threshold tuning.

Paragraphs remain the default because they match the historical SaT recipe.
Documents remain a valid engineering choice when document grouping or packing
is useful. The matched result provides no reason to expect a general quality
gain from changing units.

## Paragraph run

The mC4 paragraph model trained for 200,000 steps on the pinned
`markus583/mC4-TEST` dataset. Mean logged training loss was 0.4154, final logged
loss was 0.3428, and the rk10 run took about 26.1 hours at 1,088 samples per
second.

| Evaluation set | Languages | Macro F1 fixed | Macro F1 tuned |
| --- | ---: | ---: | ---: |
| UD | 57 | 0.9127 | 0.9354 |
| OPUS100 | 82 | 0.7807 | 0.8176 |
| Ersatz | 22 | 0.9337 | 0.9429 |

The published SaT 3L language average is 0.849. This reproduction reaches
0.832 at the fixed threshold. English is close to the published result: 0.926
across UD, OPUS100, and Ersatz, compared with 0.937 in the paper. Most of the
multilingual difference comes from OPUS100.

The run exposed a checkpoint-save bug in the Stage-1 wrapper. Its final weights
used `backbone.*` keys and lacked a model config and tokenizer. The stored mC4
checkpoint was repaired before evaluation. `wtpsplit/train/trainer.py` now
saves the unwrapped backbone, config, tokenizer, and training arguments; new
runs retain resumable checkpoints until promotion.

## Threshold behavior

The fixed Stage-1 threshold is 0.01. Roughly three quarters of evaluated
language-corpus cells prefer a tuned threshold of at least 0.02. Median tuned
thresholds are 0.032 for paragraphs and 0.034 for documents, with a cross-model
Spearman correlation of 0.91.

This calibration problem is shared by both units. Document training did not
make it worse. Corpus-specific threshold variation within a language is larger
than the difference between the two unit choices.

## Result status

| Comparison | Status |
| --- | --- |
| mC4 paragraphs vs mC4 documents | complete; near tie |
| FineWeb paragraphs vs mC4 paragraphs | pending |
| FineWeb documents vs mC4 documents | pending |
| XLM-R vs mmBERT on mC4 paragraphs | pending |
| XLM-R vs mmBERT on FineWeb paragraphs | pending |
| FineWeb scale-out | planned; corpus and weighted sampler not materialized |

An earlier mC4 document smoke used about 53 million characters and 5,000 steps.
Its 0.721 language macro F1 cannot be compared with the 51-billion-character,
200,000-step paragraph run. It only established that the document pipeline
could train and evaluate.

Raw predictions, trainer state, and model weights are run artifacts. Keep them
with the cluster run and copy compact summaries and hashes into the repository.
The selection record remains `pending_matched_runs` until every primary arm has
been evaluated under the same protocol.
