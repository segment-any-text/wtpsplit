import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial
from glob import glob
from typing import List

import numpy as np
import torch
import wandb
from datasets import load_dataset
from torch import nn
from transformers import HfArgumentParser, TrainingArguments

from wtpsplit.models import (BertCharConfig, BertCharForTokenClassification,
                             LACanineConfig, LACanineForTokenClassification)
from wtpsplit.train.evaluate import evaluate_sentence
from wtpsplit.train.trainer import Trainer
from wtpsplit.utils import Constants, LabelArgs, corrupt, get_label_dict


class Model(nn.Module):
    def __init__(
        self,
        backbone,
        loss_margin=0.5,
        use_loss_weights=False,
        do_sentence_training=True,
        do_auxiliary_training=False,
    ):
        super().__init__()
        self.backbone = backbone
        self.config = self.backbone.config

        assert loss_margin <= 0.5

        self.loss_margin = loss_margin
        self.use_loss_weights = use_loss_weights
        self.do_sentence_training = do_sentence_training
        self.do_auxiliary_training = do_auxiliary_training

    @property
    def device(self):
        return self.backbone.device

    def forward(
        self,
        input_ids,
        language_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        label_weights=None,
        **kwargs,
    ):
        reduced_attention_mask = (input_ids != 0).to(torch.long)

        output = dict(
            self.backbone.forward(
                input_ids=input_ids,
                language_ids=language_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
        )
        logits = output["logits"]

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss(reduction="none")

            losses = []

            if self.do_sentence_training:
                sentence_labels = (0.5 - self.loss_margin) + (labels == Constants.NEWLINE_INDEX + 1).to(
                    logits.dtype
                ).view(-1) * self.loss_margin * 2
                sentence_logits = logits[:, :, Constants.NEWLINE_INDEX].view(-1)

                losses.append(
                    (
                        loss_fn(
                            sentence_logits,
                            sentence_labels,
                        )
                        * (label_weights.view(-1) if label_weights is not None and self.use_loss_weights else 1)
                        * reduced_attention_mask.view(-1)
                    ).sum()
                    / reduced_attention_mask.sum()
                )

            if self.do_auxiliary_training:
                loss_fn = nn.CrossEntropyLoss()

                aux_labels = torch.where(
                    (labels == 0) | (labels == Constants.NEWLINE_INDEX + 1),
                    0,
                    labels - Constants.AUX_OFFSET,
                )
                aux_labels = torch.where(
                    reduced_attention_mask == 1,
                    aux_labels,
                    loss_fn.ignore_index,
                )

                losses.append(
                    loss_fn(
                        logits[:, :, Constants.AUX_OFFSET :].view(-1, self.config.num_labels - Constants.AUX_OFFSET),
                        aux_labels.view(-1),
                    )
                )

            loss = torch.stack(losses).sum()

            output["loss"] = loss

        return output


@dataclass
class Args:
    model_name_or_path: str
    shuffle: bool = False
    use_logits: bool = False
    is_decoder: bool = False
    use_bert: bool = False
    train_text_path: str = "data/train.parquet"
    valid_text_path: str = "data/valid.parquet"
    include_languages: List[str] = None
    eval_data_path: str = "data/eval.pth"
    num_hidden_layers: int = 1
    preprocessing_num_workers: int = 6
    block_size: int = 512
    overflow_size: int = 16
    eval_stride: int = 256
    lookahead: int = None
    loss_margin: float = 0.5
    ngram_order: int = 1
    language_adapter: str = "on"
    from_scratch: bool = False
    pack_samples: bool = False
    one_sample_per_line: bool = False
    use_loss_weights: bool = False
    do_sentence_training: bool = True
    do_auxiliary_training: bool = True
    ignore_non_hyphen: bool = False
    non_punctuation_sample_ratio: float = None
    adapter_warmup_steps: int = 0
    adapter_lr_multiplier: float = 1.0
    text_column: str = "text"


def collate_fn(batch, args, label_args, label_dict):
    all_input_ids = []
    all_labels = []
    all_language_ids = []

    all_attention_masks = []
    all_position_ids = []
    all_label_weights = []

    for sample in batch:
        input_ids = [ord(c) for c in sample[args.text_column]]
        lang = sample["lang"]

        while len(input_ids) < args.block_size + args.overflow_size:
            input_ids.append(0)

        block_ids = [0] * len(input_ids)

        input_ids, _, labels = corrupt(
            input_ids,
            block_ids,
            lang,
            label_args,
            label_dict=label_dict,
            pack_samples=args.pack_samples,
            min_length=args.block_size,
        )

        if len(input_ids) > args.block_size:
            start = np.random.randint(0, len(input_ids) - args.block_size)
            input_ids = input_ids[start : start + args.block_size]
            labels = labels[start : start + args.block_size]

        input_ids = torch.tensor(input_ids[: args.block_size], dtype=torch.long)
        labels = torch.tensor(labels[: args.block_size], dtype=torch.long)

        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        label_weights = torch.ones(args.block_size, dtype=torch.float32)
        attention_mask = (input_ids != 0).to(torch.float32)

        all_input_ids.append(input_ids)
        all_label_weights.append(label_weights)
        all_labels.append(labels)
        all_language_ids.append(Constants.LANG_CODE_TO_INDEX[lang])

        all_attention_masks.append(attention_mask)
        all_position_ids.append(position_ids)

    out = {
        "input_ids": torch.stack(all_input_ids, 0),
        "attention_mask": torch.stack(all_attention_masks, 0),
        "position_ids": torch.stack(all_position_ids, 0),
        "language_ids": torch.tensor(all_language_ids, dtype=torch.long),
        "label_weights": torch.stack(all_label_weights, 0),
        "labels": torch.stack(all_labels, 0),
    }

    return out


def main():
    parser = HfArgumentParser([Args, TrainingArguments, LabelArgs])

    if sys.argv[1].endswith(".json"):
        (args, training_args, label_args) = parser.parse_json_file(sys.argv[1])
        wandb_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    else:
        (args, training_args, label_args) = parser.parse_args_into_dataclasses()
        wandb_name = None

    config = LACanineConfig.from_pretrained(
        args.model_name_or_path,
        raw_lookahead=args.lookahead,
        num_hidden_layers=args.num_hidden_layers,
        num_labels=Constants.AUX_OFFSET + ((1 + len(Constants.PUNCTUATION_CHARS)) if args.do_auxiliary_training else 0),
        n_languages=len(Constants.LANG_CODE_TO_INDEX),
        ngram_order=args.ngram_order,
        language_adapter=args.language_adapter,
        # upsampling kernel size > 1 is problematic for packing
        # using ks=1 doesn't allow reusing the pretrained weights
        # but if we warm it up alongside the adapters
        # there is almost no difference.
        upsampling_kernel_size=1,
    )
    if args.use_bert:
        config = BertCharConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=Constants.AUX_OFFSET
            + ((1 + len(Constants.PUNCTUATION_CHARS)) if args.do_auxiliary_training else 0),
        )
        backbone = BertCharForTokenClassification(config)
    elif args.from_scratch:
        backbone = LACanineForTokenClassification(config)
    else:
        backbone = LACanineForTokenClassification.from_pretrained(args.model_name_or_path, config=config)

    model = Model(
        backbone,
        loss_margin=args.loss_margin,
        use_loss_weights=args.use_loss_weights,
        do_sentence_training=args.do_sentence_training,
        do_auxiliary_training=args.do_auxiliary_training,
    )

    def prepare_dataset(
        path,
        num_workers=1,
        include_languages=None,
        shuffle=False,
    ):
        dataset = load_dataset("parquet", data_files=path, split="train")
        if include_languages is not None:
            include_languages = set(include_languages)

            dataset = dataset.filter(
                lambda example: example["lang"] in include_languages,
                num_proc=args.preprocessing_num_workers,
            )

        if shuffle:
            dataset = dataset.shuffle(seed=42)

        if args.ignore_non_hyphen:
            with training_args.main_process_first():
                dataset = dataset.filter(
                    lambda sample: any(c in sample[args.text_column] for c in label_args.hyphen_chars),
                    num_proc=args.preprocessing_num_workers,
                )

        if args.non_punctuation_sample_ratio is not None:
            languages_without_punctuation = {
                lang_code
                for lang_code in Constants.LANGINFO.index
                if Constants.LANGINFO.loc[lang_code, "no_punctuation"]
            }

            def drop_some_non_punctuation_samples(examples):
                include_indices = set(
                    np.where([lang_code not in languages_without_punctuation for lang_code in examples["lang"]])[0]
                )
                punctuation_indices = set(
                    i for i in np.where(examples["ends_with_punctuation"])[0] if i in include_indices
                )

                target_n_non_punct = int(
                    (len(punctuation_indices) * args.non_punctuation_sample_ratio)
                    / (1 - args.non_punctuation_sample_ratio)
                )
                n_drop = (len(include_indices) - len(punctuation_indices)) - target_n_non_punct

                out = [True for _ in range(len(examples["ends_with_punctuation"]))]

                if n_drop <= 0:
                    return out
                else:
                    drop_indices = np.random.choice(
                        list(include_indices - punctuation_indices),
                        n_drop,
                        replace=False,
                    )

                    for i in drop_indices:
                        out[i] = False

                    return out

            with training_args.main_process_first():
                dataset = dataset.filter(
                    drop_some_non_punctuation_samples,
                    batched=True,
                    batch_size=1_000_000,
                    num_proc=num_workers,
                )

        def group_texts(examples):
            all_input_blocks = []
            all_input_block_lengths = []
            all_langs = []

            def maybe_pad(text):
                if args.pack_samples:
                    padding = config.downsampling_rate - (len(text) % config.downsampling_rate)
                    if padding == config.downsampling_rate:
                        padding = 0

                    text += chr(0) * padding

                return text

            for current_lang in set(examples["lang"]):
                lang_texts = [
                    maybe_pad(text)
                    for text, lang in zip(examples[args.text_column], examples["lang"])
                    if lang == current_lang
                ]

                if args.pack_samples:
                    blocks = []
                    block_ids = []

                    current_block = ["", []]

                    for i, text in enumerate(lang_texts):
                        if len(text) > args.block_size:
                            continue

                        current_block[0] += text
                        current_block[1] += [i] * len(text)

                        if i + 1 < len(lang_texts) and len(current_block[0]) + len(lang_texts[i + 1]) > args.block_size:
                            padding = args.block_size - len(current_block[0])

                            current_block[0] += chr(0) * padding
                            current_block[1] += [i] * padding
                            blocks.append(current_block[0])
                            block_ids.append(current_block[1])

                            current_block = ["", []]

                    if len(current_block[0]) > 0:
                        padding = args.block_size - len(current_block[0])

                        current_block[0] += chr(0) * padding
                        current_block[1] += [i] * padding
                        blocks.append(current_block[0])
                        block_ids.append(current_block[1])
                else:
                    concatenated_texts = "".join(lang_texts)
                    concatenated_ids = [i for i, text in enumerate(lang_texts) for _ in text]

                    total_length = len(concatenated_texts)

                    best_length = math.ceil(total_length / args.block_size) * args.block_size + args.overflow_size
                    while best_length > total_length:
                        best_length -= args.block_size

                    if best_length < 0:
                        continue

                    concatenated_texts = concatenated_texts[:best_length]
                    concatenated_ids = concatenated_ids[:best_length]

                    blocks = [
                        concatenated_texts[i : i + args.block_size + args.overflow_size]
                        for i in range(0, best_length - args.block_size, args.block_size)
                    ]
                    block_ids = [
                        concatenated_ids[i : i + args.block_size + args.overflow_size]
                        for i in range(0, best_length - args.block_size, args.block_size)
                    ]

                block_langs = [current_lang] * len(blocks)

                all_input_blocks.extend(blocks)
                all_input_block_lengths.extend([list(Counter(ids).values()) for ids in block_ids])
                all_langs.extend(block_langs)

            return {
                args.text_column: all_input_blocks,
                "block_lengths": all_input_block_lengths,
                "lang": all_langs,
            }

        if args.do_auxiliary_training:
            assert label_args.use_auxiliary

        if args.pack_samples:
            assert not args.one_sample_per_line

        if not args.one_sample_per_line:
            with training_args.main_process_first():
                dataset = dataset.map(
                    group_texts,
                    batched=True,
                    num_proc=num_workers,
                    # a bit hacky but oh well, only drop if sentence
                    remove_columns=["ends_with_punctuation"] if args.text_column == "text" else [],
                )

        return dataset

    train_dataset = prepare_dataset(
        args.train_text_path,
        num_workers=args.preprocessing_num_workers,
        include_languages=args.include_languages,
        shuffle=args.shuffle,
    )
    valid_dataset = prepare_dataset(
        args.valid_text_path,
        num_workers=args.preprocessing_num_workers,
        include_languages=args.include_languages,
        shuffle=False,
    )

    eval_data = torch.load(
        args.eval_data_path,
    )

    def compute_metrics(trainer):
        metrics = {}
        avg_metrics = defaultdict(lambda: [])

        model = trainer._wrap_model(trainer.model, training=False)

        for lang_code, lang_data in eval_data.items():
            if args.include_languages is not None and lang_code not in args.include_languages:
                continue

            if trainer.args.process_index == 0 and args.do_sentence_training:
                for dataset_name, dataset in lang_data["sentence"].items():
                    score, _ = evaluate_sentence(
                        lang_code,
                        dataset["data"],
                        model,
                        stride=args.eval_stride,
                        block_size=args.block_size,
                        batch_size=training_args.per_device_eval_batch_size,
                    )
                    metrics[f"{lang_code}_{dataset_name}_pr_auc"] = score
                    avg_metrics[f"average_{dataset_name}_pr_auc"].append(score)

        for name, values in avg_metrics.items():
            if len(values) > 1:
                metrics[name] = np.mean(values)

        return metrics

    if "wandb" in training_args.report_to and training_args.process_index == 0:
        wandb.init(name=wandb_name, project="sentence")
        wandb.config.update(args)
        wandb.config.update(training_args)
        wandb.config.update(label_args)

        model.config.wandb_run_id = wandb.run.id

        for file in glob(os.path.join(os.path.dirname(__file__), "*.py")):
            wandb.save(os.path.abspath(file), policy="now")

    label_dict = get_label_dict(label_args)

    # needed in the trainer
    training_args.adapter_warmup_steps = args.adapter_warmup_steps
    training_args.adapter_lr_multiplier = args.adapter_lr_multiplier

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=partial(collate_fn, args=args, label_args=label_args, label_dict=label_dict),
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
