# wtpsplit🪓

Code for the paper [Where's the Point? Self-Supervised Multilingual Punctuation-Agnostic Sentence Segmentation](https://arxiv.org/abs/2305.18893) with Jonas Pfeiffer and Ivan Vulić, accepted at ACL 2023.

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
wtp.half().to("cuda")

# returns ["This is a test", "This is another test."]
wtp.split("This is a test This is another test.")

# returns an iterator yielding a lists of sentences for every text
# do this instead of calling wtp.split on every text individually for much better performance
wtp.split(["This is a test This is another test.", "And some more texts..."])

# if you're using a model with language adapters, also pass a `lang_code`
wtp.split("This is a test This is another test.", lang_code="en")

# depending on your usecase, adaptation to e.g. the Universal Dependencies style may give better results
# this always requires a language code
wtp.split("This is a test This is another test.", lang_code="en", style="ud")
```

## ONNX support

You can enable ONNX inference for the `wtp-bert-*` models:

```python
wtp = WtP("wtp-bert-mini", onnx_providers=["CUDAExecutionProvider"])
```

This requires `onnxruntime` and `onnxruntime-gpu`. It should give a good speedup on GPU!

```python
>>> from wtpsplit import WtP
>>> texts = ["This is a sentence. This is another sentence."] * 1000

# PyTorch GPU
>>> model = WtP("wtp-bert-mini")
>>> model.half().to("cuda")
>>> %timeit list(model.split(texts))
272 ms ± 16.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# onnxruntime GPU
>>> model = WtP("wtp-bert-mini", ort_providers=["CUDAExecutionProvider"])
>>> %timeit list(model.split(texts))
198 ms ± 1.36 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

Notes:
- The `wtp-canine-*` models are currently not supported with ONNX because the pooling done by CANINE is not trivial to export. Ideas to solve this are very welcome!
- This does not work with Python 3.7 because `onnxruntime` does not support the opset we need for py37.


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

This also allows changing the threshold, but inherently has higher thresholds values since it is not newline probablity anymore being thresholded:

```python
wtp.split(text, lang_code="en", style="ud", threshold=0.7)
```

To get the default threshold for a style:
```python
wtp.get_threshold("en", "ud", return_punctuation_threshold=True)
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

## Citation

Please cite `wtpsplit` as 

```
@inproceedings{minixhofer-etal-2023-wheres,
    title = "Where{'}s the Point? Self-Supervised Multilingual Punctuation-Agnostic Sentence Segmentation",
    author = "Minixhofer, Benjamin  and
      Pfeiffer, Jonas  and
      Vuli{\'c}, Ivan",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.398",
    pages = "7215--7235",
    abstract = "Many NLP pipelines split text into sentences as one of the crucial preprocessing steps. Prior sentence segmentation tools either rely on punctuation or require a considerable amount of sentence-segmented training data: both central assumptions might fail when porting sentence segmenters to diverse languages on a massive scale. In this work, we thus introduce a multilingual punctuation-agnostic sentence segmentation method, currently covering 85 languages, trained in a self-supervised fashion on unsegmented text, by making use of newline characters which implicitly perform segmentation into paragraphs. We further propose an approach that adapts our method to the segmentation in a given corpus by using only a small number (64-256) of sentence-segmented examples. The main results indicate that our method outperforms all the prior best sentence-segmentation tools by an average of 6.1{\%} F1 points. Furthermore, we demonstrate that proper sentence segmentation has a point: the use of a (powerful) sentence segmenter makes a considerable difference for a downstream application such as machine translation (MT). By using our method to match sentence segmentation to the segmentation used during training of MT models, we achieve an average improvement of 2.3 BLEU points over the best prior segmentation tool, as well as massive gains over a trivial segmenter that splits text into equally-sized blocks.",
}
```

## Acknowledgments

Ivan Vulić is supported by a personal Royal Society University Research Fellowship ‘Inclusive and Sustainable Language Technology for a Truly Multilingual World’ (no 221137; 2022–). Research supported with Cloud TPUs from Google’s TPU Research Cloud (TRC). We thank Christoph Minixhofer for advice in the initial stage of this project. We also thank Sebastian Ruder and Srini Narayanan for helpful feedback on a draft of the paper.

## Previous Version

*This repository previously contained `nnsplit`, the precursor to `wtpsplit`. You can still use the `nnsplit` branch (or the `nnsplit` PyPI releases) for the old version, however, this is highly discouraged and not maintained! Please let me know if you have a usecase which `nnsplit` can solve but `wtpsplit` can not.*
