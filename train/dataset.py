import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
from labeler import get_default_labeler
import random


class SplitDataset(data.Dataset):
    def __init__(self, text_dataset, min_len, max_len, max_pad):
        self.text_dataset = text_dataset
        self.labeler = get_default_labeler()
        self.min_len = min_len
        self.max_len = max_len
        self.max_pad = max_pad

    def __len__(self):
        return len(self.text_dataset)

    def __getitem__(self, idx):
        original_text = self.text_dataset[idx]
        length = random.randint(self.min_len, self.max_len)

        first_bound = -self.max_pad
        second_bound = len(original_text) - length + self.max_pad

        # very unlikely edge case, just add one
        if first_bound == second_bound:
            second_bound += 1

        start = random.randint(
            min(first_bound, second_bound), max(first_bound, second_bound)
        )
        end = start + length

        ids, label = self.labeler.label(
            original_text[max(start, 0) : min(end, len(original_text))]
        )

        for _ in range(start, 0):
            ids.insert(0, 0)
            label.insert(0, [0] * len(self.labeler.tokenizers))

        for _ in range(len(original_text), end):
            ids.append(0)
            label.append([0] * len(self.labeler.tokenizers))

        return torch.tensor(ids), torch.tensor(label)

    @staticmethod
    def collate_fn(batch):
        inputs, labels = zip(*batch)

        inputs = pad_sequence(inputs, batch_first=True)
        labels = pad_sequence(labels, batch_first=True)

        return inputs, labels


if __name__ == "__main__":
    from text_data import MemoryMapDataset
    from tqdm.auto import tqdm
    from torch.utils import data

    text_data = MemoryMapDataset("texts.txt", "slices.pkl")
    dataset = SplitDataset(text_data, 500, 800, 20)
    loader = data.DataLoader(
        dataset,
        collate_fn=SplitDataset.collate_fn,
        shuffle=False,
        batch_size=128,
        num_workers=6,
    )

    for batch in tqdm(loader):
        continue
