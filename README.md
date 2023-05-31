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

