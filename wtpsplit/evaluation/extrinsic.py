from dataclasses import dataclass

import pandas as pd
import regex as re
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, HfArgumentParser, pipeline

import sacrebleu
import skops.io as sio
from datasets import load_dataset
from wtpsplit.evaluation import ERSATZ_LANGUAGES, ersatz_sentencize, our_sentencize, preprocess_sentence
from wtpsplit.utils import Constants


@dataclass
class Args:
    model_path: str
    device: str = "cuda"
    clfs_path: str = Constants.CACHE_DIR / "main_3layers_h0_clfs.pkl"
    paragraph_length: int = 10


def mt_eval(
    sentence_model,
    lang_code,
    paragraph_length,
    clf,
    features,
    threshold_transformed,
    threshold_newline,
):
    dset_name = Constants.LANGINFO.loc[lang_code, "opus100"]
    dset_langs = dset_name.split("-")
    target_lang = dset_langs[0] if dset_langs[1] == lang_code else dset_langs[1]

    dset = load_dataset("opus100", dset_name)

    try:
        mt_pipeline = pipeline(
            "translation",
            model=f"Helsinki-NLP/opus-mt-{lang_code}-{target_lang}",
            device=0,
        )
    except OSError:
        return (None, None, None, None)

    all_sentences = [x[lang_code] for x in dset["test"]["translation"]]
    all_targets = [x[target_lang] for x in dset["test"]["translation"]]

    separator = Constants.SEPARATORS[lang_code]

    current_paragraph = []
    current_target = []

    all_preds_ground_truth = []
    all_preds_ersatz = []
    all_preds_ours = []
    all_preds_naive = []
    all_preds_no_sentencize = []

    all_target_texts = []

    bar = tqdm(total=len(all_sentences))

    while len(all_sentences) > 0:
        while len(current_paragraph) < paragraph_length and len(all_sentences) > 0:
            current_paragraph.append(preprocess_sentence(all_sentences.pop(0)))
            current_target.append(all_targets.pop(0))

            bar.update(1)

        paragraph_text = separator.join(current_paragraph)
        target_text = separator.join(current_target)

        preds_ground_truth = separator.join(
            [x["translation_text"] for x in mt_pipeline(current_paragraph, truncation=True)]
        )

        if lang_code in ERSATZ_LANGUAGES:
            preds_ersatz = separator.join(
                [
                    x["translation_text"]
                    for x in mt_pipeline(
                        [x.strip() for x in ersatz_sentencize(lang_code, paragraph_text)],
                        truncation=True,
                    )
                ]
            )
        else:
            preds_ersatz = None

        preds_ours = separator.join(
            [
                x["translation_text"]
                for x in mt_pipeline(
                    [
                        x.strip()
                        for x in our_sentencize(
                            sentence_model,
                            lang_code,
                            paragraph_text,
                            clf,
                            features,
                            threshold_transformed,
                            threshold_newline,
                        )
                    ],
                    truncation=True,
                )
            ]
        )
        preds_no_sentencize = mt_pipeline(paragraph_text, truncation=True)[0]["translation_text"]
        if Constants.LANGINFO.loc[lang_code, "no_whitespace"]:
            naive_sentences = [
                paragraph_text[
                    int(i * len(paragraph_text) / len(current_paragraph)) : int(
                        (i + 1) * len(paragraph_text) / len(current_paragraph)
                    )
                ]
                for i in range(len(current_paragraph))
            ]
        else:
            words = [x for x in re.split(r"(\S+\s+)", paragraph_text) if len(x) > 0]
            naive_sentences = [
                "".join(
                    words[
                        int(i * len(words) / len(current_paragraph)) : int(
                            (i + 1) * len(words) / len(current_paragraph)
                        )
                    ]
                )
                for i in range(len(current_paragraph))
            ]

        assert "".join(naive_sentences) == paragraph_text
        preds_naive = separator.join(
            [x["translation_text"] for x in mt_pipeline([x.strip() for x in naive_sentences], truncation=True)]
        )

        all_preds_naive.append(preds_naive)
        all_preds_ground_truth.append(preds_ground_truth)
        if preds_ersatz is not None:
            all_preds_ersatz.append(preds_ersatz)
        all_preds_ours.append(preds_ours)
        all_preds_no_sentencize.append(preds_no_sentencize)

        all_target_texts.append(target_text)

        current_paragraph = []
        current_target = []

    bleu = sacrebleu.BLEU()

    return (
        bleu.corpus_score(all_preds_no_sentencize, [all_target_texts]).score,
        bleu.corpus_score(all_preds_ground_truth, [all_target_texts]).score,
        bleu.corpus_score(all_preds_naive, [all_target_texts]).score,
        bleu.corpus_score(all_preds_ersatz, [all_target_texts]).score if len(all_preds_ersatz) > 0 else None,
        bleu.corpus_score(all_preds_ours, [all_target_texts]).score,
    )


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    model = AutoModelForTokenClassification.from_pretrained(args.model_path).to(args.device)

    clfs = sio.load(open(args.clfs_path, "rb"), ["numpy.float64", "numpy.float32"])

    langs = [
        "ar",
        "cs",
        "de",
        "en",
        "es",
        "fi",
        "hi",
        "ja",
        "ka",
        "lv",
        "pl",
        "th",
        "xh",
        "zh",
    ]

    results = []

    for lang_code in tqdm(langs):
        results.append(
            (
                lang_code,
                *mt_eval(
                    model,
                    lang_code,
                    args.paragraph_length,
                    *clfs[lang_code]["opus100"],
                ),
            ),
        )
        print(results[-1])

    pd.DataFrame(results, columns=["lang", "no_sent", "ground_truth", "naive", "ersatz", "ours"]).to_csv(
        "results.csv", index=False
    )
