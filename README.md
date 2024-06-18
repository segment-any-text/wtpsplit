# Segment any Text: Robust, Efficient and Adaptable Sentence Segmentation

Code for the paper [Segment Any Text: A Universal Approach for Robust, Efficient and Adaptable Sentence Segmentation](TODO) by Markus Frohmann, Igor Sterner, Benjamin Minixhofer, Ivan Vulić and Markus Schedl.

This repository contains `segment-any-text`, a package for robust, efficient and adaptable sentence segmentation across 85 languages, as well as the code and configs to reproduce the **state-of-the-art** results in 8 distinct corpora and 85 languages demonstrated in our paper.

![System Figure](./system-fig.png)


## Installation

```bash
pip install segment-any-text
```

## Usage

```python
from sat import SaT

sat = SaT("sat-3l")
# optionally run on GPU for better performance
# also supports TPUs via e.g. wtp.to("xla:0"), in that case pass `pad_last_batch=True` to wtp.split
sat.half().to("cuda")

# returns ["This is a test", "This is another test."]
sat.split("This is a test This is another test.")

# returns an iterator yielding a lists of sentences for every text
sat.split(["This is a test This is another test.", "And some more texts..."])

# use our '-sm' models for general sentence segmentation tasks
sat_sm = SaT("sat-3l-sm")
# this will be especially better for noisy text
sat.split("this is a test this is another test")
# returns ["this is a test", "this is another test"]

# use trained lora modules for strong adaptation to language & domain/style
sat.split("This is a test This is another test.", lang_code="en", style="ud")
```


## Available Models

If you need a general sentence segmentation model, use `-sm` models (e.g., `sat-3l-sm`)
For speed-sensitive applications, we recommend 3-layer models (`sat-3l` and `sat-3l-sm`). They provide a good tradeoff between speen and performance.
The best (and largest) models are our 12-layer models: `sat-12l` and `sat-12l-sm`.
## TODO TODO TODO
<!-- 
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

The scores are macro-average F1 score across all available datasets for "English", and macro-average F1 score across all datasets and languages for "Multilingual". "adapted" means adapation via LoRA; check out the paper for details. 

For comparison, here's the English scores of some other tools:

| Model                                                                      |    English Score
|:-----------------------------------------------------------------------|-----:|
| SpaCy (sentencizer) | 86.8 |
| PySBD | 69.8 |
| SpaCy (dependency parser) | 93.1 |
| Ersatz | 91.6 |
| Punkt (`nltk.sent_tokenize`) | 92.5 | -->

Note that this library also supports previous [`WtP`](https://arxiv.org/abs/2305.18893) models.
You can use them in essentially the same way as `SaT`models:

```python
from sat import WtP

wtp = WtP("wtp-bert-mini")
# similar functionality as for SaT models
wtp.split("This is a test This is another test.")
```

For more details on WtP and reproduction details, see the `wtpsplit` branch.

## Paragraph Segmentation

Since SaT are trained to predict newline probablity, they can segment text into paragraphs in addition to sentences.

```python
# returns a list of paragraphs, each containing a list of sentences
# adjust the paragraph threshold via the `paragraph_threshold` argument.
sat.split(text, do_paragraph_segmentation=True)
```

## Adaptation


SaT can be domain- and style-adapted via LoRA. We provide trained LoRA modules for Universal Dependencies, OPUS100, Ersatz, and TED (i.e., ASR-style transcribed speecjes) sentence styles in 81 languages for `sat-3l`and `sat-12l`. Additionally, we provide LoRA modules for legal documents (laws and judgements) in 6 languages, code-switching in 4 language pairs, and tweets in 3 languages. For details, we refer to our [paper](TODO).

We also provided verse segmentation modules for 16 genres for `sat-12-no-limited-lookahead`.

Load LoRA modules like this:
```python

# requires both lang_code and style_or_domain
# for available ones, check the <model_repository>/loras folder
sat_lora = SaT("sat-3l", style_or_domain="ud", language="en")
sat_lora.split("Hello this is a test But this is different now Now the next one starts looool")
# now for a highly distinct domain
sat_lora_distinct = SaT("sat-12l", style_or_domain="code-switching", language="es-en")
sat_lora_distinct.split("in the morning over there cada vez que yo decía algo él me decía algo")
```

You can also freely adapt the segmentation threshold, with a higher threshold leading to more conservative segmentation:
```python

sat.split("This is a test This is another test.", threshold=0.4)
# works similarly for lora; but thresholds are higher
sat_lora.split("Hello this is a test But this is different now Now the next one starts looool", threshold=0.7)
```
<!-- 
#### WtP Adaptation

WtP can adapt to the Universal Dependencies, OPUS100 or Ersatz corpus segmentation style in many languages by punctuation adaptation (*preferred*) or threshold adaptation.

##### Punctuation Adaptation

```python
# this requires a `lang_code`
# check the WtP paper or `wtp.mixtures` for supported styles
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

##### Threshold Adaptation
```python
threshold = wtp.get_threshold("en", "ud")

wtp.split(text, threshold=threshold)
``` -->

## Advanced Usage

### Get the newline or sentence boundary probabilities for a text:

```python
# returns newline probabilities (supports batching!)
sat.predict_proba(text)
```

### Load a SaT model in [HuggingFace `transformers`](https://github.com/huggingface/transformers):

```python
# import library to register the custom models 
import wtpsplit
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained("segment-any-text/sat-3l-sm") # or some other model name; see https://huggingface.co/segment-any-text
```

### Adapt to your own corpus via LoRA
Our models can be efficiently adapted via LoRA in a powerful way. Only 10-100 training segmented training sentences should already improve performance considerably. To do so:

Clone the repository and install requirements:

```
git clone https://github.com/segment-any-text/segment-any-text
cd segment-any-text
pip install -e .
pip install -r requirements.txt
cd adapters
pip install -e .
cd ..
```

Create data in this format:
```python
import torch

torch.save(
    {
        "language_code": {
            "sentence": {
                "dummy-dataset": {
                    "meta": {
                        "train_data": ["train sentence 1", "train sentence 2"],
                    },
                    "data": [
                        "test sentence 1",
                        "test sentence 2",
                    ]
                }
            }
        }
    },
    "dummy-dataset.pth"
)
```

Create/adapt config; provide base model via `model_name_or_path` and training data .pth via ' text_path`:


`configs/lora/lora_dummy_config.json`

Train LoRA:
```
python3 wtpsplit/train/train_lora.py configs/lora/lora_dummy_config.json
```

Once training is done, provide your saved module's path to SaT:
```python

sat_lora_adapted = SaT("model-used", lora_path="dummy_lora_path")
sat_lora_adapted.split("Some domains-specific or styled text")
```

Adjust the dataset name, language and model in the above to your needs.


## Reproducing the paper

`configs/` contains the configs for the runs from the paper for base and sm models as well as LoRA modules. Launch training for each of them like this:

```
python3 wtpsplit/train/train.py configs/<config_name>.json
python3 wtpsplit/train/train_sm.py configs/<config_name>.json
python3 wtpsplit/train/train_lora.py configs/<config_name>.json
```

In addition:
- `wtpsplit/data_acquisition` contains the code for obtaining evaluation data and raw text from the mC4 corpus.
- `wtpsplit/evaluation` contains the code for:
  - evaluation (i.e. sentence segmentation results) via `intrinsic.py`. 
  - short-sequence evaluation (i.e. sentence segmentation results for pairs/k-mers of sentences) via `intrinsic_pairwise.py`. 
  - LLM baseline evaluation (`llm_sentence.py`), legal baseline evaluation (`legal_baselines.py`)
  - baseline (PySBD, nltk, etc.) evaluation results in `intrinsic_baselines.py` and `intrinsic_baselines_multi.py`
  - Raw results in JSON format are also in `evaluation_results/`
  - Statistical significane testing code and results ara in `stat_tests/`
  - punctuation annotation experiments in `punct_annotation.py` and `punct_annotation_wtp.py` (WtP only)
  - extrinsic evaluation on Machine Translation in `extrinsic.py` (WtP only)
  
Ensure to install packages from `requirements.txt` beforehand.
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

For details, we refer to our [paper](TODO).

## Citation

If you find our `segment-any-text` useful, please kindly cite our paper:
```
@inproceedings{TODO,}
```
If you use WtP models, cite:
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
    pages = "7215--7235"
}
```

## Acknowledgments

This research was funded in whole or in part by the Austrian Science Fund (FWF): P36413, P33526, and DFH-23, and by the State of Upper Austria and the Federal Ministry of Education, Science, and Research, through grants LIT-2021-YOU-215. In addition, Ivan Vulic and Benjamin Minixhofer ´have been supported through the Royal Society University Research Fellowship ‘Inclusive and Sustainable Language Technology for a Truly Multilingual World’ (no 221137) awarded to Ivan Vulic.´ This research has also been supported with Cloud TPUs from Google’s TPU Research Cloud (TRC). This work was also supported by compute credits
from a Cohere For AI Research Grant, these grants are designed to support academic partners conducting research with the goal of releasing scientific artifacts and data for good projects. We also thank Simone Teufel for fruitful discussions.


## Previous Version

*This repository previously contained `nnsplit` and `wtpsplit`, the precursors to `segment-any-text`. We still support all functionality of `wtpsplit`. Moreover, you can still use the `nnsplit` branch (or the `nnsplit` PyPI releases) for the old version, however, this is highly discouraged and not maintained! Please let us know if you have a usecase which `nnsplit` can solve but `segment-any-test` can not.*

## Final Words
We hope this repo is useful. For any questions, please create an issue or send an email to markus.frohmann@gmail.com, and I will get back to you as soon as possible.