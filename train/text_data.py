import re
from xml.etree import ElementTree
from lxml.etree import iterparse
from pathlib import Path
from tqdm.auto import tqdm
import pickle
import mmap
from torch.utils import data


def _fast_iter(context):
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


def xml_dump_iter(xml_dump_path, min_text_length, max_text_length):
    for p in _fast_iter(iterparse(str(xml_dump_path), tag="p")):
        if not (min_text_length <= len(p) <= max_text_length):
            continue

        yield p


class MemoryMapDataset(data.Dataset):
    @staticmethod
    def iterator_to_text_and_slices(iterator, text_path, slice_path, max_n_texts=None):
        slices = []
        offset = 0
        i = 0

        with open(text_path, "wb") as f:
            for text in tqdm(iterator):
                text = text.encode("utf-8")

                slices.append((offset, offset + len(text)))
                f.write(text)

                i += 1
                offset += len(text)

                if max_n_texts is not None and i >= max_n_texts:
                    break

        with open(slice_path, "wb") as f:
            pickle.dump(slices, f)

    def __init__(self, text_path: Path, slice_path: Path):
        self.text_path = text_path
        self.slices = pickle.load(open(slice_path, "rb"))

        self.file = open(self.text_path, "r+b")
        self.mm = mmap.mmap(self.file.fileno(), 0)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        start, end = self.slices[idx]
        return self.mm[start:end].decode("utf-8")


if __name__ == "__main__":
    xml_iter = xml_dump_iter(
        "../../../train_data/dewiki-20180920-corpus.xml",
        min_text_length=300,
        max_text_length=5000,
    )

    MemoryMapDataset.iterator_to_text_and_slices(xml_iter, "texts.txt", "slices.pkl")
