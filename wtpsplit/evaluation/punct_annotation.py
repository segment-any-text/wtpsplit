import math
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import wandb
from datasets import load_dataset
from sklearn.metrics import f1_score
from torch import nn
from tqdm.auto import tqdm
from transformers import (AutoModel, AutoModelForTokenClassification,
                          AutoTokenizer, CanineTokenizer, HfArgumentParser,
                          Trainer, TrainingArguments)

from wtpsplit.models import (LACanineConfig, LACanineForTokenClassification,
                             LACanineModel)
from wtpsplit.utils import Constants


@dataclass
class Args:
    name: str
    lang: str
    model_path: str = None
    baseline: str = None
    block_size: int = 256
    stride: int = 32
    use_lang_adapter: bool = False
    use_deep_punctuation: bool = False


class DeepPunctuation(nn.Module):
    def __init__(self, backbone, freeze_bert=False, lstm_dim=-1):
        super(DeepPunctuation, self).__init__()
        self.output_dim = 4
        self.bert_layer = backbone
        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        bert_dim = backbone.config.hidden_size
        if lstm_dim == -1:
            hidden_size = bert_dim
        else:
            hidden_size = lstm_dim
        self.lstm = nn.LSTM(
            input_size=bert_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
        )
        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=self.output_dim)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # (B, N, E) -> (B, N, E)
        x = self.bert_layer(input_ids, attention_mask=attention_mask)[0]
        # (B, N, E) -> (N, B, E)
        x = torch.transpose(x, 0, 1)
        x, (_, _) = self.lstm(x)
        # (N, B, E) -> (B, N, E)
        x = torch.transpose(x, 0, 1)
        x = self.linear(x)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(x.view(-1, 4), labels.view(-1))

        return {"logits": x, "loss": loss}


def to_strided(input_ids, labels, offset_mapping, tokenizer, stride, block_size, add_sep_cls):
    if add_sep_cls:
        block_size -= 2
        stride -= 2

    input_id_windows = []
    label_windows = []
    offset_mapping_windows = []

    start_positions = [i * stride for i in range(len(input_ids) // stride - block_size // stride)]
    if start_positions[-1] + block_size != len(input_ids):
        start_positions[-1] = len(input_ids) - block_size

    for start in start_positions:
        if add_sep_cls:
            input_id_windows.append(
                [tokenizer.cls_token_id] + input_ids[start : start + block_size] + [tokenizer.sep_token_id]
            )
            label_windows.append([0] + labels[start : start + block_size] + [0])
            offset_mapping_windows.append([(0, 0)] + offset_mapping[start : start + block_size] + [(0, 0)])
        else:
            input_id_windows.append(input_ids[start : start + block_size])
            label_windows.append(labels[start : start + block_size])
            offset_mapping_windows.append(offset_mapping[start : start + block_size])

    return input_id_windows, label_windows, offset_mapping_windows


def process(data, tokenizer, stride, block_size, add_sep_cls, fix_space, return_chars=False):
    char_labels = []

    label_dict = {
        "O": 0,
        "QUESTION": 1,
        "PERIOD": 2,
        "COMMA": 3,
    }

    text = ""

    for i, row in enumerate(data["text"]):
        token, label = row.split("\t")

        if not fix_space or (i > 0 and token not in {"'m", "n't", "'ll", "'s"}):
            text += " "
            char_labels.append(0)

        text += token
        char_labels.extend([0] * len(token))
        char_labels[-1] = label_dict[label]

    if isinstance(tokenizer, CanineTokenizer):
        encoding = tokenizer(text, add_special_tokens=False)

        assert encoding["input_ids"] == [ord(c) for c in text]
        offset_mapping = [(i, i + 1) for i in range(len(text))]
    else:
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offset_mapping = encoding["offset_mapping"]

    labels = [char_labels[x[1] - 1] for x in offset_mapping]

    if return_chars:
        return {"labels": char_labels}

    input_id_windows, label_windows, offset_mapping_windows = to_strided(
        encoding["input_ids"],
        labels,
        offset_mapping,
        tokenizer,
        stride,
        block_size,
        add_sep_cls,
    )

    return {
        "input_ids": input_id_windows,
        "labels": label_windows,
        "offset_mapping": offset_mapping_windows,
    }


def load_iwslt(path, tokenizer, args, fix_space=True):
    dataset = load_dataset("text", data_files=str(path), split="train")
    dataset = dataset.map(
        partial(
            process,
            tokenizer=tokenizer,
            stride=args.stride,
            block_size=args.block_size,
            add_sep_cls=args.add_sep_cls,
            fix_space=fix_space,
        ),
        batched=True,
        batch_size=len(dataset),
        remove_columns=["text"],
    )

    char_dataset = load_dataset("text", data_files=str(path), split="train")
    char_labels = char_dataset.map(
        partial(
            process,
            tokenizer=tokenizer,
            stride=args.stride,
            block_size=args.block_size,
            add_sep_cls=args.add_sep_cls,
            fix_space=fix_space,
            return_chars=True,
        ),
        batched=True,
        batch_size=len(char_dataset),
        remove_columns=["text"],
    )["labels"]

    if args.use_lang_adapter:
        dataset = dataset.add_column("language_ids", [Constants.LANG_CODE_TO_INDEX[args.lang]] * len(dataset))

    return dataset, np.array(char_labels)


def compute_metrics(_, test_dataset, test_char_labels):
    global trainer

    model = trainer._wrap_model(trainer.model, training=False)
    batch_size = trainer.args.per_device_eval_batch_size

    if trainer.args.process_index == 0:
        char_preds = np.zeros((len(test_char_labels), 4))
        char_counts = np.zeros((len(test_char_labels)))

        n_batches = math.ceil(len(test_dataset) / batch_size)
        for i in tqdm(range(n_batches)):
            batch = test_dataset[(i * batch_size) : (i + 1) * batch_size]
            while len(batch["input_ids"]) < batch_size:
                batch["input_ids"].append([0] * len(batch["input_ids"][-1]))
                batch["labels"].append([-100] * len(batch["input_ids"][-1]))
                batch["offset_mapping"].append([(0, 0)] * len(batch["input_ids"][-1]))

            with torch.no_grad():
                output = model(
                    input_ids=torch.tensor(batch["input_ids"], device=model.device),
                    labels=torch.tensor(batch["labels"], device=model.device),
                )

                batch_logits = output["logits"].cpu().numpy()
                for logits, offsets in zip(batch_logits, batch["offset_mapping"]):
                    for l, (start, end) in zip(logits, offsets):
                        if start != end:
                            char_preds[end - 1] += l
                            char_counts[end - 1] += 1

        mask = test_char_labels != -100

        comma, period, question = (
            f1_score(char_preds[mask].argmax(-1) == 3, test_char_labels[mask] == 3),
            f1_score(char_preds[mask].argmax(-1) == 2, test_char_labels[mask] == 2),
            f1_score(char_preds[mask].argmax(-1) == 1, test_char_labels[mask] == 1),
        )

        return {
            "comma_f1": comma,
            "period_f1": period,
            "question_f1": question,
            "overall_f1": np.mean([comma, period, question]),
        }
    else:
        return {}


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


def main():
    (args, training_args) = HfArgumentParser([Args, TrainingArguments]).parse_args_into_dataclasses()

    model_class = AutoModel if args.use_deep_punctuation else AutoModelForTokenClassification
    our_model_class = LACanineModel if args.use_deep_punctuation else LACanineForTokenClassification

    if args.baseline:
        assert not args.use_lang_adapter

        model = model_class.from_pretrained(args.baseline, num_labels=4)
        tokenizer = AutoTokenizer.from_pretrained(args.baseline)
        args.add_sep_cls = True
    else:
        config = LACanineConfig.from_pretrained(
            args.model_path,
            num_labels=4,
            language_adapter="on" if args.use_lang_adapter else "off",
        )
        model = our_model_class.from_pretrained(args.model_path, config=config, ignore_mismatched_sizes=True)
        if args.use_lang_adapter:
            model.set_language_adapters(args.lang)

        tokenizer = CanineTokenizer.from_pretrained("google/canine-s")
        args.add_sep_cls = False

    if args.use_deep_punctuation:
        model = DeepPunctuation(model)

    if args.lang == "en":
        train_dataset, _ = load_iwslt(
            Constants.ROOT_DIR / "data" / "external" / "punctuation_annotation" / "en" / "train2012",
            tokenizer=tokenizer,
            args=args,
        )
        test_dataset, test_char_labels = load_iwslt(
            Constants.ROOT_DIR / "data" / "external" / "punctuation_annotation" / "en" / "test2011",
            tokenizer=tokenizer,
            args=args,
        )
    else:
        train_dataset, _ = load_iwslt(
            Constants.ROOT_DIR / "data" / "external" / "punctuation_annotation" / "bn" / "train",
            tokenizer=tokenizer,
            args=args,
        )
        test_dataset, test_char_labels = load_iwslt(
            Constants.ROOT_DIR / "data" / "external" / "punctuation_annotation" / "bn" / "test_ref",
            tokenizer=tokenizer,
            args=args,
        )

    if "wandb" in training_args.report_to and training_args.process_index == 0:
        wandb.init(project="punct_annotation", name=args.name)
        wandb.config.update(args)
        wandb.config.update(training_args)

        # model.config.wandb_run_id = wandb.run.id

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=partial(
            compute_metrics,
            test_dataset=test_dataset,
            test_char_labels=test_char_labels,
        ),
    )

    globals().update({"trainer": trainer})

    trainer.train()


if __name__ == "__main__":
    main()
