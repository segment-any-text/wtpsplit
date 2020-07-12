from torch.utils import data
import numpy as np
from labeler import remove_last_punct
from tqdm.auto import tqdm


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
                    sentence = remove_last_punct(sentence)

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
