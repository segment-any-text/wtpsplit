import subprocess
import unicodedata
import os

import numpy as np
import regex as re
import torch
from sklearn import linear_model
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score

from wtpsplit.extract import extract
from wtpsplit.utils import Constants, indices_to_sentences, lang_code_to_lang, reconstruct_sentences


def preprocess_sentence(sentence):
    # right-to-left-mark
    sentence = sentence.replace(chr(8207), "")

    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", sentence.lstrip("-").strip()))


def get_labels(lang_code, sentences, after_space=True):
    separator = Constants.SEPARATORS[lang_code]
    text = separator.join(sentences)

    true_end_indices = np.cumsum(np.array([len(s) for s in sentences])) + np.arange(1, len(sentences) + 1) * len(
        separator
    )
    # no space after last
    true_end_indices[-1] -= len(separator)

    if not after_space:
        true_end_indices -= len(separator) + 1

    labels = np.zeros(len(text) + 1)
    labels[true_end_indices] = 1

    return labels


def evaluate_sentences(lang_code, sentences, predicted_sentences):
    separator = Constants.SEPARATORS[lang_code]

    text = separator.join(sentences)

    assert len(text) == len("".join(predicted_sentences))

    labels = get_labels(lang_code, sentences)

    predicted_end_indices = np.cumsum(np.array([len(s) for s in predicted_sentences]))
    predictions = np.zeros_like(labels)
    predictions[predicted_end_indices] = 1

    return f1_score(labels, predictions), {
        "recall": recall_score(labels, predictions),
        "precision": precision_score(labels, predictions),
    }


def train_mixture(lang_code, original_train_x, train_y, n_subsample=None, features=None):
    original_train_x = torch.from_numpy(original_train_x).float()

    train_y = train_y[:-1]

    if original_train_x.shape[1] > Constants.AUX_OFFSET:
        if features is not None:
            train_x = original_train_x[:, features]
        else:
            train_x = original_train_x

        train_x = train_x.float()

        clf = linear_model.LogisticRegression(max_iter=10_000, random_state=0)
        clf.fit(train_x, train_y)
        preds = clf.predict_proba(train_x)[:, 1]

        p, r, t = precision_recall_curve(train_y, preds)
        f1 = 2 * p * r / (p + r + 1e-6)
        best_threshold_transformed = t[f1.argmax()]
    else:
        clf = None
        best_threshold_transformed = None

    p, r, t = precision_recall_curve(train_y, torch.sigmoid(original_train_x[:, Constants.NEWLINE_INDEX]))
    f1 = 2 * p * r / (p + r + 1e-6)
    best_threshold_newline = t[f1.argmax()]

    return clf, features, best_threshold_transformed, best_threshold_newline


def evaluate_mixture(
    lang_code,
    test_x,
    true_sentences,
    clf,
    features,
    threshold_transformed,
    threshold_newline,
):
    test_x = torch.from_numpy(test_x)
    text = Constants.SEPARATORS[lang_code].join(true_sentences)

    predicted_indices_newline = np.where(
        torch.sigmoid(test_x[..., Constants.NEWLINE_INDEX].float()).numpy() > threshold_newline
    )[0]

    if clf is not None:
        if features is not None:
            test_x = test_x[:, features]

        test_x = test_x.float()

        probs = clf.predict_proba(test_x)[:, 1]
        predicted_indices_transformed = np.where(probs > threshold_transformed)[0]
    else:
        predicted_indices_transformed = None

    score_newline, info_newline = evaluate_sentences(
        lang_code,
        true_sentences,
        reconstruct_sentences(text, indices_to_sentences(text, predicted_indices_newline)),
    )

    if predicted_indices_transformed is None:
        return (
            score_newline,
            None,
            {"info_newline": info_newline, "info_transformed": None},
        )

    score_transformed, info_transformed = evaluate_sentences(
        lang_code,
        true_sentences,
        reconstruct_sentences(text, indices_to_sentences(text, predicted_indices_transformed)),
    )

    return (
        score_newline,
        score_transformed,
        {
            "info_newline": info_newline,
            "info_transformed": info_transformed,
        },
    )


def our_sentencize(
    sentence_model,
    lang_code,
    text,
    clf,
    features,
    threshold_transformed,
    threshold_newline,
    newline_only=False,
    block_size=512,
    stride=64,
    batch_size=32,
):
    logits = extract(
        [text],
        sentence_model,
        lang_code=lang_code,
        stride=stride,
        block_size=block_size,
        batch_size=batch_size,
        pad_last_batch=False,
        use_hidden_states=False,
        verbose=False,
    )[0]

    predicted_indices_newline = np.where(
        torch.sigmoid(logits[..., Constants.NEWLINE_INDEX].float()).numpy() > threshold_newline
    )[0]

    if features is not None:
        x = logits[:, features]
    else:
        x = logits

    x = x.float()

    probs = clf.predict_proba(x)[:, 1]
    predicted_indices_transformed = np.where(probs > threshold_transformed)[0]

    if newline_only:
        return reconstruct_sentences(text, indices_to_sentences(text, predicted_indices_newline))
    else:
        return reconstruct_sentences(text, indices_to_sentences(text, predicted_indices_transformed))


# baselines

ERSATZ_LANGUAGES = {
    "ar",
    "cs",
    "de",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "gu",
    "hi",
    "iu",
    "ja",
    "kk",
    "km",
    "lt",
    "lv",
    "pl",
    "ps",
    "ro",
    "ru",
    "ta",
    "tr",
    "zh",
}


class LanguageError(ValueError):
    pass


def ersatz_sentencize(
    lang_code,
    text,
    infile="notebooks/data/tmp/in.tmp",
    outfile="notebooks/data/tmp/out.tmp",
):
    if lang_code not in ERSATZ_LANGUAGES:
        raise LanguageError(f"ersatz does not support {lang_code}")
    
    # check if infile parent dir exists, if not, create it
    if not os.path.exists(os.path.dirname(infile)):
        os.makedirs(os.path.dirname(infile))
    # check if outfile parent dir exists, if not, create it
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
        
    open(infile, "w").write(text)

    subprocess.check_output(
        f"cat {infile} | ersatz --quiet -m default-multilingual > {outfile}",
        shell=True,
    )

    return reconstruct_sentences(text, open(outfile).readlines())


def pysbd_sentencize(lang_code, text):
    import pysbd

    try:
        return reconstruct_sentences(text, pysbd.Segmenter(language=lang_code, clean=False).segment(text))
    except ValueError:
        raise LanguageError(f"pysbd does not support {lang_code}")


SPACY_LANG_TO_DP_MODEL = {
    "ca": "ca_core_news_sm",
    "zh": "zh_core_web_sm",
    "hr": "hr_core_news_sm",
    "da": "da_core_news_sm",
    "nl": "nl_core_news_sm",
    "en": "en_core_web_sm",
    "fi": "fi_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "el": "el_core_news_sm",
    "it": "it_core_news_sm",
    "ja": "ja_core_news_sm",
    "ko": "ko_core_news_sm",
    "lt": "lt_core_news_sm",
    "mk": "mk_core_news_sm",
    "nb": "nb_core_news_sm",
    "pl": "pl_core_news_sm",
    "pt": "pt_core_news_sm",
    "ro": "ro_core_news_sm",
    "ru": "ru_core_news_sm",
    "es": "es_core_news_sm",
    "sv": "sv_core_news_sm",
    "uk": "uk_core_news_sm",
}


def spacy_sent_sentencize(lang_code, text):
    import spacy

    try:
        nlp = spacy.blank(lang_code)
        nlp.add_pipe("sentencizer")

        if lang_code == "ja":
            # spacy uses SudachiPy for japanese, which has a length limit:
            # https://github.com/WorksApplications/sudachi.rs/blob/c7d20b22c68bb3f6585351847ae91bc9c7a61ec5/sudachi/src/input_text/buffer/mod.rs#L124-L126
            # so we need to chunk the input and sentencize the chunks separately
            chunksize = 10_000
            chunks = []
            for i in range(0, len(text), chunksize):
                chunks.append(text[i : i + chunksize])

            assert sum(len(c) for c in chunks) == len(text)
            return reconstruct_sentences(text, [str(s) for c in chunks for s in nlp(c).sents])

        return reconstruct_sentences(text, list([str(s) for s in nlp(text).sents]))
    except ImportError:
        raise LanguageError(f"spacy_sent does not support {lang_code}")


def spacy_dp_sentencize(lang_code, text):
    import spacy

    try:
        nlp = spacy.load(SPACY_LANG_TO_DP_MODEL[lang_code], disable=["ner"])

        if lang_code == "ja":
            # spacy uses SudachiPy for japanese, which has a length limit:
            # https://github.com/WorksApplications/sudachi.rs/blob/c7d20b22c68bb3f6585351847ae91bc9c7a61ec5/sudachi/src/input_text/buffer/mod.rs#L124-L126
            # so we need to chunk the input and sentencize the chunks separately
            chunksize = 10_000
            chunks = []
            for i in range(0, len(text), chunksize):
                chunks.append(text[i : i + chunksize])

            assert sum(len(c) for c in chunks) == len(text)
            return reconstruct_sentences(text, [str(s) for c in chunks for s in nlp(c).sents])

        return reconstruct_sentences(text, list([str(s) for s in nlp(text).sents]))
    except KeyError:
        raise LanguageError(f"spacy_dp does not support {lang_code}")


def punkt_sentencize(lang_code, text):
    from nltk.tokenize import sent_tokenize

    try:
        return reconstruct_sentences(text, sent_tokenize(text, language=lang_code_to_lang(lang_code).lower()))
    except LookupError:
        raise LanguageError(f"punkt does not support {lang_code}")
