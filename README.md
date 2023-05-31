# wtpsplit🪓

Code for the paper [Where's the Point? Self-Supervised Multilingual Punctuation-Agnostic Sentence Segmentation](https://arxiv.org/abs/2305.18893) accepted at ACL 2023.

This repository contains `wtpsplit`, a package for robust and adaptible sentence segmentation across 85 languages, as well as the code and configs to reproduce the experiments in the paper.

## Installation

```bash
pip install wtpsplit
```

## Usage

```python
from wtpsplit import WtP

wtp = WtP("wtp-bert-mini")
# optionally run on GPU for better performance
# also supports TPUs via e.g. wtp.to("xla:0"), in that case pass `pad_last_batch=True` to wtp.split
wtp.to("cuda")

# returns ["This is a test", "This is another test."]
wtp.split("This is a test This is another test.")

# returns an iterator yielding a lists of sentences for every text
# do this instead of calling wtp.split on every text individually for much better performance
wtp.split(["This is a test This is another test.", "And some more texts..."])
```

## Available Models

Pro tips: I recommend `wtp-bert-mini` for speed-sensitive applications, otherwise `wtp-canine-s-12l`. The `*-no-adapters` models provide a good tradeoff between speed and performance. You should *probably not* use `wtp-bert-tiny`.

| Model                                                                      |    English Score |    English Score<br>(adapted) |    Multilingual Score |    Multilingual Score<br>(adapted) |
|:-----------------------------------------------------------------------|-----:|-----:|-----:|-----:|
| [wtp-bert-tiny](https://huggingface.co/benjamin/wtp-bert-tiny)                | 83.8 | 91.9 | 79.5 | 88.6 |
| [wtp-bert-mini](https://huggingface.co/benjamin/wtp-bert-mini)                | 91.8 | 95.9 | 84.3 | 91.3 |
| [wtp-canine-s-1l](https://huggingface.co/benjamin/wtp-canine-s-1l)              | 94.5 | 96.5 | 86.7 | 92.8 |
| [wtp-canine-s-1l-no-adapters](https://huggingface.co/benjamin/wtp-canine-s-1l-no-adapters)  | 93.1 | 96.4 | 85.1 | 91.8 |
| [wtp-canine-s-3l](https://huggingface.co/benjamin/wtp-canine-s-3l)              | 94.4 | 96.8 | 86.7 | 93.4 |
| [wtp-canine-s-3l-no-adapters](https://huggingface.co/benjamin/wtp-canine-s-3l-no-adapters)  | 93.8 | 96.4 | 86   | 92.3 |
| [wtp-canine-s-6l](https://huggingface.co/benjamin/wtp-canine-s-6l)              | 94.5 | 97.1 | 87   | 93.6 |
| [wtp-canine-s-6l-no-adapters](https://huggingface.co/benjamin/wtp-canine-s-6l-no-adapters)  | 94.4 | 96.8 | 86.4 | 92.8 |
| [wtp-canine-s-9l](https://huggingface.co/benjamin/wtp-canine-s-9l)              | 94.8 | 97   | 87.7 | 93.8 |
| [wtp-canine-s-9l-no-adapters](https://huggingface.co/benjamin/wtp-canine-s-9l-no-adapters)  | 94.3 | 96.9 | 86.6 | 93   |
| [wtp-canine-s-12l](https://huggingface.co/benjamin/wtp-canine-s-12l)             | 94.7 | 97.1 | 87.9 | 94   |
| [wtp-canine-s-12l-no-adapters](https://huggingface.co/benjamin/wtp-canine-s-12l-no-adapters) | 94.5 | 97   | 87.1 | 93.2 |

The scores are macro-average F1 score across all available datasets for "English", and macro-average F1 score across all datasets and languages for "Multilingual". "adapted" means adapation via WtP Punct; check out the paper for details. 

For comparison, here's the English scores of some other tools:

| Model                                                                      |    English Score
|:-----------------------------------------------------------------------|-----:|
| SpaCy (sentencizer) | 86.8 |
| PySBD | 69.8 |
| SpaCy (dependency parser) | 93.1 |
| Ersatz | 91.6 |
| Punkt (`nltk.sent_tokenize`) | 92.5 |

### Paragraph Segmentation

Since WtP models are trained to predict newline probablity, they can segment text into paragraphs in addition to sentences.

```python
# returns a list of paragraphs, each containing a list of sentences
# adjust the paragraph threshold via the `paragraph_threshold` argument.
wtp.split(text, do_paragraph_segmentation=True)
```

### Adaptation

WtP can adapt to the Universal Dependencies, OPUS100 or Ersatz corpus segmentation style in many languages by punctuation adaptation (*preferred*) or threshold adaptation.

#### Punctuation Adaptation

```python
# this requires a `lang_code`
# check the paper or `wtp.mixtures` for supported styles
wtp.split(text, lang_code="en", style="ud")
```

#### Threshold Adaptation
```python
threshold = wtp.get_threshold("en", "ud")

wtp.split(text, threshold=threshold)
```

### Advanced Usage

Get the newline or sentence boundary probabilities for a text:

```python
# returns newline probabilities (supports batching!)
wtp.predict_proba(text)

# returns sentence boundary probabilities for the given style
wtp.predict_proba(text, lang_code="en", style="ud")
```

Load a WtP model in [HuggingFace `transformers`](https://github.com/huggingface/transformers):

```python
# import wtpsplit to register the custom models 
# (character-level BERT w/ hash embeddings and canine with language adapters)
import wtpsplit
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained("benjamin/wtp-bert-mini") # or some other model name
```

## Reproducing the paper

`configs/` contains the configs for the runs from the paper. We trained on a TPUv3-8. Launch training like this:

```
python wtpsplit/train/train.py configs/<config_name>.json
```

In addition:
- `wtpsplit/data_acquisition` contains the code for obtaining evaluation data and raw text from the mC4 corpus.
- `wtpsplit/evaluation` contains the code for:
  - intrinsic evaluation (i.e. sentence segmentation results) via `intrinsic.py`. The raw intrinsic results in JSON format are also at `evaluation_results/`
  - extrinsic evaluation on Machine Translation in `extrinsic.py`
  - baseline (PySBD, nltk, etc.) intrinsic evaluation in `intrinsic_baselines.py`
  - punctuation annotation experiments in `punct_annotation.py` and `punct_annotation_wtp.py`

## ONNX support (experimental)

You can enable ONNX inference for the `wtp-bert-*` models:

```python
wtp = WtP("wtp-bert-mini", onnx_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```

This requires `onnxruntime` and `onnxruntime-gpu`.

However, on my hardware, it *did not* produce a speedup over PyTorch. The embeddings in ONNX inference still have to be computed using PyTorch because hash embeddings are supported by ONNX, so the moving around of tensors might cause it to be slower.

The `wtp-canine-*` models are currently not supported with ONNX because the pooling done by CANINE is not trivial to export. 

Ideas to solve this (and the hash embeddings problem) are very welcome!

## Supported Languages

| iso | Name                   |
|:----|:-----------------------|
| af  | Afrikaans              |
| am  | Amharic                |
| ar  | Arabic                 |
| az  | Azerbaijani            |
| be  | Belarusian             |
| bg  | Bulgarian              |
| bn  | Bengali                |
| ca  | Catalan                |
| ceb | Cebuano                |
| cs  | Czech                  |
| cy  | Welsh                  |
| da  | Danish                 |
| de  | German                 |
| el  | Greek                  |
| en  | English                |
| eo  | Esperanto              |
| es  | Spanish                |
| et  | Estonian               |
| eu  | Basque                 |
| fa  | Persian                |
| fi  | Finnish                |
| fr  | French                 |
| fy  | Western Frisian        |
| ga  | Irish                  |
| gd  | Scottish Gaelic        |
| gl  | Galician               |
| gu  | Gujarati               |
| ha  | Hausa                  |
| he  | Hebrew                 |
| hi  | Hindi                  |
| hu  | Hungarian              |
| hy  | Armenian               |
| id  | Indonesian             |
| ig  | Igbo                   |
| is  | Icelandic              |
| it  | Italian                |
| ja  | Japanese               |
| jv  | Javanese               |
| ka  | Georgian               |
| kk  | Kazakh                 |
| km  | Central Khmer          |
| kn  | Kannada                |
| ko  | Korean                 |
| ku  | Kurdish                |
| ky  | Kirghiz                |
| la  | Latin                  |
| lt  | Lithuanian             |
| lv  | Latvian                |
| mg  | Malagasy               |
| mk  | Macedonian             |
| ml  | Malayalam              |
| mn  | Mongolian              |
| mr  | Marathi                |
| ms  | Malay                  |
| mt  | Maltese                |
| my  | Burmese                |
| ne  | Nepali                 |
| nl  | Dutch                  |
| no  | Norwegian              |
| pa  | Panjabi                |
| pl  | Polish                 |
| ps  | Pushto                 |
| pt  | Portuguese             |
| ro  | Romanian               |
| ru  | Russian                |
| si  | Sinhala                |
| sk  | Slovak                 |
| sl  | Slovenian              |
| sq  | Albanian               |
| sr  | Serbian                |
| sv  | Swedish                |
| ta  | Tamil                  |
| te  | Telugu                 |
| tg  | Tajik                  |
| th  | Thai                   |
| tr  | Turkish                |
| uk  | Ukrainian              |
| ur  | Urdu                   |
| uz  | Uzbek                  |
| vi  | Vietnamese             |
| xh  | Xhosa                  |
| yi  | Yiddish                |
| yo  | Yoruba                 |
| zh  | Chinese                |
| zu  | Zulu                   |

## Previous Version

*This repository previously contained `nnsplit`, the precursor to `wtpsplit`. You can still use the `nnsplit` branch (or the `nnsplit` PyPI releases) for the old version, however, this is highly discouraged and not maintained! Please let me know if you have a usecase which `nnsplit` can solve but `wtpsplit` can not.*
