from pathlib import Path
import random
import re
from xml.etree import ElementTree
import numpy as np
from lxml.etree import iterparse
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.utils import data
from fastai.train import Learner, DataBunch
from .tokenizer import SoMaJoTokenizer
from .utils import text_to_id
from .defaults import CUT_LENGTH
from .models import Network

REMOVE_DOT_CHANCE = 0.5
LOWERCASE_START_CHANCE = 0.5
MIN_LENGTH = 600
N_CUTS = 4


def label_paragraph(paragraph, tokenizer):
    tokenized_p = tokenizer.split([paragraph])[0]

    text = ""
    labels = []

    for sentence in tokenized_p:
        for i, token in enumerate(sentence):
            text_to_append = token.text + token.whitespace

            if (
                token.text == "."
                and i == len(sentence) - 1
                and random.random() < REMOVE_DOT_CHANCE
            ):
                text_to_append = token.whitespace
                if len(text_to_append) > 0 and len(labels) > 1:
                    labels[-1][0] = 0.0

            if i == 0 and random.random() < LOWERCASE_START_CHANCE:
                text_to_append = token.text.lower() + token.whitespace

            for _ in range(len(text_to_append)):
                labels.append([0.0, 0.0])

            if len(labels) > 0:
                labels[-1][0] = 1.0

            text += text_to_append

        labels[-1][1] = 1.0

    return text, labels


def generate_data(paragraph, tokenizer, min_length, n_cuts, cut_length):
    if len(paragraph) < min_length:
        return [], []

    p_text, p_labels = label_paragraph(paragraph, tokenizer)
    assert len(p_text) == len(p_labels)

    inputs = [[] for _ in range(n_cuts)]
    labels = [[] for _ in range(n_cuts)]

    for j in range(n_cuts):
        start = random.randint(0, len(p_text))

        for k in range(cut_length):
            if start + k >= len(p_text):
                inputs[j].append(0)
                labels[j].append([0.0, 0.0])
            else:
                inputs[j].append(text_to_id(p_text[start + k]))
                labels[j].append(p_labels[start + k])

    return inputs, labels


def fast_iter(context):
    for event, elem in context:
        text = ElementTree.tostring(elem, encoding="utf8").decode("utf-8")
        text = re.sub(r"(<h>(.*?)<\/h>)", "\n", text)
        text = re.sub(r"<.*?>", "", text)
        text = text.strip()
        yield text

        # It's safe to call clear() here because no descendants will be
        # accessed
        elem.clear()
        # Also eliminate now-empty references from the root node to elem
        for ancestor in elem.xpath("ancestor-or-self::*"):
            while ancestor.getprevious() is not None:
                parent = ancestor.getparent()

                if parent is not None:
                    del parent[0]
                else:
                    break


def prepare_data(
    corpus,
    language,
    max_n_sentences,
    data_directory=None,
    remove_dot_chance=REMOVE_DOT_CHANCE,
    lowercase_start_chance=LOWERCASE_START_CHANCE,
    min_length=MIN_LENGTH,
    n_cuts=N_CUTS,
    cut_length=CUT_LENGTH,
):
    if data_directory is not None:
        data_directory = Path(data_directory)
        data_directory.mkdir(exist_ok=True, parents=True)

    all_sentences = torch.zeros([max_n_sentences, cut_length], dtype=torch.uint8)
    all_labels = torch.zeros([max_n_sentences, cut_length, 2], dtype=torch.bool)

    tokenizer = SoMaJoTokenizer(language)
    bar = tqdm(total=max_n_sentences)

    i = 0
    for paragraph in fast_iter(iterparse(str(corpus), tag="p")):
        text, labels = generate_data(
            paragraph, tokenizer, min_length, n_cuts, cut_length
        )

        length = min(len(text), max_n_sentences - i)

        if length > 0:
            all_sentences[i : i + length] = torch.tensor(
                text[:length], dtype=torch.uint8
            )
            all_labels[i : i + length] = torch.tensor(labels[:length], dtype=torch.bool)

        i = i + length

        if i == max_n_sentences:
            break

        bar.update(length)

    if i < max_n_sentences:
        all_sentences = all_sentences[:i]
        all_labels = all_labels[:i]

    if data_directory is not None:
        torch.save(all_sentences, data_directory / "all_sentences.pt")
        torch.save(all_labels, data_directory / "all_labels.pt")

    return all_sentences, all_labels


def loss(inputs, targets):
    return F.binary_cross_entropy_with_logits(inputs, targets.float())


def train_from_tensors(
    all_sentences, all_labels, valid_percent=0.1, batch_size=128, n_epochs=10
):
    n_valid = int(len(all_sentences) * valid_percent)

    permutation = np.random.permutation(np.arange(len(all_sentences)))
    valid_idx, train_idx = permutation[:n_valid], permutation[n_valid:]

    train_dataset = data.TensorDataset(all_sentences[train_idx], all_labels[train_idx])
    valid_dataset = data.TensorDataset(all_sentences[valid_idx], all_labels[valid_idx])

    model = Network()

    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False
    )
    valid_loader = data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=False
    )

    databunch = DataBunch(train_dl=train_loader, valid_dl=valid_loader)
    learn = Learner(databunch, model, loss_func=loss)

    if torch.cuda.is_available():
        learn = learn.to_fp16()

    learn.fit_one_cycle(n_epochs)

    return learn


def train_from_directory(data_directory, *args, **kwargs):
    all_sentences = torch.load(Path(data_directory) / "all_sentences.pt")
    all_labels = torch.load(Path(data_directory) / "all_labels.pt")

    return train_from_tensors(all_sentences, all_labels, *args, **kwargs)
