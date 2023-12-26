import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial
from glob import glob
from typing import List
import random

import numpy as np
import torch
import wandb
from datasets import load_dataset
from datasets.download import DownloadConfig
from torch import nn
from transformers import HfArgumentParser, TrainingArguments, AutoTokenizer, set_seed
from torchinfo import summary
from tokenizers import AddedToken

from wtpsplit.models import (
    BertCharConfig,
    BertCharForTokenClassification,
    LACanineConfig,
    LACanineForTokenClassification,
    SubwordXLMConfig,
    SubwordXLMForTokenClassification,
)
from wtpsplit.train.evaluate import evaluate_sentence
from wtpsplit.train.trainer import Trainer
from wtpsplit.utils import Constants, LabelArgs, corrupt, get_label_dict, get_subword_label_dict

# TODO: set logger (see ScaLearn?)

# TODO: double-check checkpointing and saving (also to txt)

# os.environ["PJRT_DEVICE"] = "None"


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
        if position_ids is not None:
            reduced_attention_mask = (input_ids != 0).to(torch.long)
        else:
            # XXX: 1 is pad token id
            reduced_attention_mask = (input_ids != 1).to(torch.long)

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

            # main (newline prediction) objective
            if self.do_sentence_training:
                # label smoothing
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

            # auxiliary (punctuation prediction) objective
            if self.do_auxiliary_training:
                loss_fn = nn.CrossEntropyLoss()

                # exclude newline and no labels
                aux_labels = torch.where(
                    (labels == 0) | (labels == Constants.NEWLINE_INDEX + 1),
                    0,
                    labels - Constants.AUX_OFFSET,
                )
                # exclude reduced_attention_mask tokens from labels
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
    # TODO: adapt to HF Hub
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

    # NEW PARAMS
    use_subwords: bool = False


def collate_fn(batch, args, label_args, label_dict, tokenizer):
    all_input_ids = []
    all_labels = []
    all_language_ids = []

    all_attention_masks = []
    all_position_ids = []
    all_label_weights = []

    for sample in batch:
        # subword-level
        if args.use_subwords:
            input_ids = sample["input_ids"]
        # char-level
        else:
            input_ids = [ord(c) for c in sample["input_ids"]]
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
            tokenizer=tokenizer if args.use_subwords else None,
        )
        
        # if input_ids[0] != tokenizer.cls_token_id:
        #     print(input_ids)
        #     print(len(input_ids))
        #     print(tokenizer.cls_token_id)
        #     raise ValueError("CLS token not first token")
        # if input_ids[-1] != tokenizer.sep_token_id:
        #     print(input_ids)
        #     print(len(input_ids))
        #     print(tokenizer.sep_token_id)
        #     raise ValueError("SEP token not last token")

        if len(input_ids) > args.block_size:
            if tokenizer:
                # always include CLS
                start = np.random.randint(0, len(input_ids) - args.block_size)
                if start != 0:
                    # this removes the CLS token
                    # -1 removes the SEP token, for sure
                    input_ids = [tokenizer.cls_token_id] + input_ids[start : start + args.block_size - 2]
                    labels = [0] + labels[start : start + args.block_size - 2]
                else:
                    input_ids = input_ids[start : start + args.block_size - 1]
                    labels = labels[start : start + args.block_size - 1]
                # always include SEP
                if input_ids[-1] != tokenizer.sep_token_id:
                    input_ids = input_ids + [tokenizer.sep_token_id]
                    labels = labels + [0]
            else:
                start = np.random.randint(0, len(input_ids) - args.block_size)
                input_ids = input_ids[start : start + args.block_size]
                labels = labels[start : start + args.block_size]

        input_ids = torch.tensor(input_ids[: args.block_size], dtype=torch.long)
        labels = torch.tensor(labels[: args.block_size], dtype=torch.long)
        # if input_ids[-1] != tokenizer.sep_token_id:
        #     print(input_ids)
        #     print(tokenizer.sep_token_id)
        #     print(labels)
        #     raise ValueError("SEP token not last token")
        # if input_ids[0] != tokenizer.cls_token_id:
        #     print(input_ids)
        #     print(tokenizer.cls_token_id)
        #     print(labels)
        #     raise ValueError("CLS token not first token")
        # TODO: check this - why does it occur in train split?
        # if (input_ids == tokenizer.cls_token_id).sum() != 1:
        #     print(input_ids)
        #     print(tokenizer.cls_token_id)
        #     print(labels)
        #     raise ValueError("CLS token not unique")

        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        label_weights = torch.ones(args.block_size, dtype=torch.float32)
        if tokenizer:
            attention_mask = (input_ids != tokenizer.pad_token_id).to(torch.float32)
        else:
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
        "position_ids": torch.stack(all_position_ids, 0) if not args.use_subwords else None,  # safer
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
        
    set_seed(training_args.seed)

    num_labels = Constants.AUX_OFFSET + ((1 + len(Constants.PUNCTUATION_CHARS)) if args.do_auxiliary_training else 0)
    if args.use_subwords:
        if args.from_scratch:
            config = SubwordXLMConfig.from_pretrained(
                args.model_name_or_path,
                num_hidden_layers=args.num_hidden_layers,
                num_labels=num_labels,
            )
            backbone = SubwordXLMForTokenClassification(config)
            
        else:
            config = SubwordXLMConfig.from_pretrained(
                args.model_name_or_path,
                num_hidden_layers=args.num_hidden_layers,
                num_labels=num_labels,
            )
            backbone = SubwordXLMForTokenClassification(config)

        backbone.config.base_model = args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        # needed since we create labels in collate_fn based on tokens
        # TODO: problematic for <UNK> tokens!
        tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})

    else:
        tokenizer = None
        config = LACanineConfig.from_pretrained(
            args.model_name_or_path,
            raw_lookahead=args.lookahead,
            num_hidden_layers=args.num_hidden_layers,
            num_labels=num_labels,
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
                num_labels=num_labels,
            )
            backbone = BertCharForTokenClassification(config)
        elif args.from_scratch:
            backbone = LACanineForTokenClassification(config)
        else:
            backbone = LACanineForTokenClassification.from_pretrained(
                args.model_name_or_path, ignore_mismatched_sizes=True, config=config
            )

    model = Model(
        backbone,
        loss_margin=args.loss_margin,
        use_loss_weights=args.use_loss_weights,
        do_sentence_training=args.do_sentence_training,
        do_auxiliary_training=args.do_auxiliary_training,
    )

    with training_args.main_process_first():
        print(summary(model, depth=4))
        # backbone.push_to_hub("markus583/xlm-token-untrained", private=True)

    def prepare_dataset(
        num_workers=1,
        include_languages=None,
        shuffle=False,
        split="train",
    ):
        with training_args.main_process_first():
            dlconf = DownloadConfig(cache_dir="/home/Markus/.cache/huggingface/datasets")
            dataset = load_dataset("markus583/mC4-TEST", split=split, download_config=dlconf)
        # optional: delete downloaded dataset, it is stored in cache_dir now (but we delete it later)
        # ~40GB on disk
        # os.system("rm -rf /home/Markus/.cache/huggingface/datasets")

        if include_languages is not None:
            include_languages = set(include_languages)

            dataset = dataset.filter(
                lambda example: example["lang"] in include_languages,
                num_proc=args.preprocessing_num_workers,
            )

        if shuffle:
            dataset = dataset.shuffle(seed=42)

        # very likely not relevant / used only for the compound part
        if args.ignore_non_hyphen:
            with training_args.main_process_first():
                dataset = dataset.filter(
                    lambda sample: any(c in sample[args.text_column] for c in label_args.hyphen_chars),
                    num_proc=args.preprocessing_num_workers,
                )

        # "punctuation-specific sampling" in the paper
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
        

        def tokenize_texts(examples):
            # do not return CLS and SEP token here
            # there should only be 1 of these per block later, not multiple
            # we still can't use return_special_tokens=False since we need the \n token later for the labels
            tokenized = tokenizer(examples[args.text_column], verbose=False)
            return {"input_ids": [example[1:-1] for example in tokenized["input_ids"]]}

        # similar to group_texts in huggingface's run_clm.py / run_mlm.py: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
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
                if not args.use_subwords:
                    lang_texts = [
                        maybe_pad(text)
                        for text, lang in zip(examples[args.text_column], examples["lang"])
                        if lang == current_lang
                    ]
                else:
                    # only retain current_lang examples (all columns)
                    lang_subwords = [
                        subwords
                        for subwords, lang in zip(examples["input_ids"], examples["lang"])
                        if lang == current_lang
                    ]

                # pack_samples used for the compound part, so irrelevant
                if args.pack_samples:
                    if args.use_subwords:
                        raise NotImplementedError
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
                    if not args.use_subwords:
                        concatenated_texts = "".join(lang_texts)
                        concatenated_ids = [i for i, text in enumerate(lang_texts) for _ in text]
                    else:
                        # concatenate lists
                        concatenated_texts = [item for sublist in lang_subwords for item in sublist]
                        concatenated_ids = [i for i, subwords in enumerate(lang_subwords) for _ in subwords]

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
                "input_ids": all_input_blocks,
                "block_lengths": all_input_block_lengths,
                "lang": all_langs,
            }

        if args.do_auxiliary_training:
            assert label_args.use_auxiliary

        if args.pack_samples:
            assert not args.one_sample_per_line

        if args.use_subwords:
            with training_args.main_process_first():
                dataset = dataset.map(
                    tokenize_texts,
                    batched=True,
                    num_proc=num_workers,
                    remove_columns=[args.text_column],
                )
                
        if split == "train":
            with training_args.main_process_first():
                for root, dirs, files in os.walk(os.environ.get("HF_DATASETS_CACHE")):
                    for file in files:
                        if file.startswith("m_c4-test-train"):
                            print(f"Removing {os.path.join(root, file)}")
                            os.remove(os.path.join(root, file))

        if not args.one_sample_per_line:
            with training_args.main_process_first():
                dataset = dataset.map(
                    group_texts,
                    batched=True,
                    num_proc=num_workers,
                    # a bit hacky but oh well, only drop if sentence
                    remove_columns=["ends_with_punctuation"]  # FIXME: needed for char-based args.text_column dropping
                    if args.text_column == "text"
                    else [],
                )

        return dataset

    valid_dataset = prepare_dataset(
        num_workers=args.preprocessing_num_workers,
        include_languages=args.include_languages,
        shuffle=False,
        split="valid",
    )
    train_dataset = prepare_dataset(
        num_workers=args.preprocessing_num_workers,
        include_languages=args.include_languages,
        shuffle=args.shuffle,
        split="train",
    )

    # print some samples from the dataset
    count = 0
    while count < 5:
        index = random.choice(range(len(train_dataset)))
        sample = train_dataset[index]

        if sample.get('lang') == "de":
            print(f"Sample {index} of the training set: {sample}.")
            if tokenizer:
                print(tokenizer.decode(sample["input_ids"]))
            print()
            count += 1

    # dataset we use is in cached now
    # m_c4 files are test/valid splits of already downloaded data
    # ~80GB deleted, ~63 GB left in cache/RAM (cache-* files)
    # with training_args.main_process_first():
    #     for root, dirs, files in os.walk(os.environ.get("HF_DATASETS_CACHE")):
    #         for file in files:
    #             if file.startswith("m_c4"):
    #                 print(f"Removing {os.path.join(root, file)}")
    #                 os.remove(os.path.join(root, file))

    eval_data = torch.load(
        args.eval_data_path,
    )

    def compute_metrics(trainer):
        metrics = {}
        avg_metrics = defaultdict(lambda: [])

        model = trainer._wrap_model(trainer.model, training=False)

        for lang_code, lang_data in eval_data.items():  # TODO: tqdm integration
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

    label_dict = get_subword_label_dict(label_args, tokenizer) if args.use_subwords else get_label_dict(label_args)

    # needed in the trainer
    training_args.adapter_warmup_steps = args.adapter_warmup_steps
    training_args.adapter_lr_multiplier = args.adapter_lr_multiplier

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=partial(
            collate_fn,
            args=args,
            label_args=label_args,
            label_dict=label_dict,
            tokenizer=tokenizer if args.use_subwords else None,
        ),
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
