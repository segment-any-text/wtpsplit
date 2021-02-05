from torch.utils import data
import numpy as np
from labeler import remove_last_punct, get_model
from tqdm.auto import tqdm
import click
import spacy
import pandas as pd


class OpenSubtitlesDataset(data.Dataset):
    def __init__(self, source_file, max_lines=None):
        self.sentences = []
        for i, line in tqdm(enumerate(open(source_file))):
            if max_lines is not None and i >= max_lines:
                break

            self.sentences.append(self.clean(line))

        self.sentences = [x for x in self.sentences if x is not None]

    def clean(self, line, min_length=2):
        clean = line.lstrip("-").strip()
        return clean if len(clean) >= min_length else None

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


class Evaluator:
    def __init__(
        self,
        dataset,
        remove_end_punct_prob,
        lower_start_prob,
        punctuation,
        lengths=[2, 3, 4],
        seed=1234,
    ):
        gen = np.random.RandomState(seed)
        self.sentence_groups = []

        end = 0
        bar = tqdm(total=len(dataset))

        while end < len(dataset):
            start = end
            end = min(start + gen.choice(lengths), len(dataset))

            sentence_group = []

            for i in range(start, end):
                sentence = dataset[i]

                if gen.random() < remove_end_punct_prob:
                    sentence = remove_last_punct(sentence, punctuation)

                if gen.random() < lower_start_prob:
                    sentence = sentence[0].lower() + sentence[1:]

                # whitespace which joins sentences is expected to be part of the previous sentence
                if i < end - 1:
                    sentence += " "

                sentence_group.append(sentence)

            self.sentence_groups.append(sentence_group)
            bar.update(len(sentence_group))

        self.texts = ["".join(group) for group in self.sentence_groups]

    def evaluate(self, split_fn):
        correct = np.full(len(self.texts), False, dtype=np.bool)

        predicted_groups = split_fn(self.texts)
        for i, (predicted_group, group) in enumerate(
            zip(predicted_groups, self.sentence_groups)
        ):
            if len(predicted_group) != len(group):
                continue

            for (a, b) in zip(predicted_group, group):
                if a != b:
                    continue

            correct[i] = True

        return correct


class NNSplitInterface:
    def __init__(self, splitter):
        self.splitter = splitter

    def split(self, texts):
        out = []
        for split in self.splitter.split(texts):
            out.append([str(x) for x in split])

        return out


class SpacyInterface:
    def __init__(self, name, use_sentencizer, batch_size=1000):
        if use_sentencizer:
            nlp = get_model(name)
            nlp.add_pipe("sentencizer")
        else:
            try:
                nlp = spacy.load(name, disable=["tagger", "ner"])
            except OSError:
                nlp = None

        self.nlp = nlp
        self.batch_size = batch_size

    def split(self, texts):
        out = []

        if self.nlp is not None:
            for doc in self.nlp.pipe(texts, batch_size=self.batch_size):
                sentences = []

                for sent in doc.sents:
                    sentences.append("".join([x.text + x.whitespace_ for x in sent]))

                out.append(sentences)

        return out


@click.command()
@click.option("--subtitle_path", help="Path to the OPUS OpenSubtitles raw text.")
@click.option("--spacy_model", help="Name of the spacy model to compare against.")
@click.option("--nnsplit_path", help="Path to the .onnx NNSplit model to use.")
@click.option("--punctuation", help="Which characters to consider punctuation.", default=".?!")
def evaluate(subtitle_path, spacy_model, nnsplit_path, punctuation):
    # nnsplit must be installed to evaluate
    from nnsplit import NNSplit

    print("Evaluating..")

    dataset = data.Subset(
        OpenSubtitlesDataset(subtitle_path, 1_000_000), np.arange(100_000)
    )
    targets = {
        "NNSplit": NNSplitInterface(
            NNSplit(nnsplit_path, use_cuda=True, batch_size=2 ** 7)
        ),
        "Spacy (Tagger)": SpacyInterface(spacy_model, use_sentencizer=False),
        "Spacy (Sentencizer)": SpacyInterface(spacy_model, use_sentencizer=True),
    }

    eval_setups = {
        "Clean": (0.0, 0.0),
        "Partial punctuation": (0.5, 0.0),
        "Partial case": (0.0, 0.5),
        "Partial punctuation and case": (0.5, 0.5),
        "No punctuation and case": (1.0, 1.0),
    }

    result = {}
    preds = {}

    for eval_name, (remove_punct_prob, lower_start_prob) in eval_setups.items():
        result[eval_name] = {}
        evaluator = Evaluator(dataset, remove_punct_prob, lower_start_prob, punctuation)

        for target_name, interface in targets.items():
            correct = evaluator.evaluate(interface.split)
            preds[f"{eval_name}_{target_name}"] = {
                "samples": evaluator.texts,
                "correct": correct,
            }
            result[eval_name][target_name] = correct.mean()

    result = pd.DataFrame.from_dict(result).T
    print(result)
    print(result.to_markdown())


if __name__ == "__main__":
    evaluate()
