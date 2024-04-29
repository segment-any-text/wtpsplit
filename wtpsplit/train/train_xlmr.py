import logging
import math
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial
from glob import glob
from typing import List, Optional

import datasets
import numpy as np
import torch
import torch_xla.core.xla_model as xm
import transformers
from datasets import load_dataset
from datasets.download import DownloadConfig
from torchinfo import summary
from tqdm.auto import tqdm
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed

import wandb
from wtpsplit.models import (
    SubwordXLMConfig,
    SubwordXLMForTokenClassification,
)
from wtpsplit.train.evaluate import evaluate_sentence, evaluate_sentence_kmers, evaluate_sentence_pairwise
from wtpsplit.train.trainer import Trainer
from wtpsplit.train.utils import Model, cleanup_cache_files
from wtpsplit.utils import Constants, LabelArgs, corrupt_training, get_label_dict, get_subword_label_dict
from wtpsplit.tokenization_utils import pack_sentences

logger = logging.getLogger(__name__)


# os.environ["PJRT_DEVICE"] = "None"


def setup_logging(training_args: transformers.TrainingArguments) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process the small summary:
    logger.warning(
        (
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f"distributed training: {training_args.local_rank != -1}, 16-bits training: {training_args.fp16}"
        )
    )
    # logger.info(f"Training/evaluation parameters {training_args}")


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
    eval_data_path: str = "data/all_data_24_04.pth"
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
    aux_training_weight: float = 1.0
    ignore_non_hyphen: bool = False
    non_punctuation_sample_ratio: float = None
    adapter_warmup_steps: int = 0
    adapter_lr_multiplier: float = 1.0
    text_column: str = "text"

    # NEW PARAMS
    use_subwords: bool = False
    threshold: float = 0.01
    lookahead_split_layers: Optional[int] = None


def collate_fn(batch, args, label_args, label_dict, tokenizer, add_lang_ids: bool = False):
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

        newline_label_indices = sample["labels"]
        newline_labels = [1 if i in newline_label_indices else 0 for i in range(len(input_ids))]

        while len(input_ids) < args.block_size + args.overflow_size:
            input_ids.append(tokenizer.pad_token_id)
            newline_labels.append(0)

        block_ids = [0] * len(input_ids)

        input_ids, _, labels = corrupt_training(
            input_ids,
            block_ids,
            newline_labels,
            lang,
            label_args,
            label_dict=label_dict,
            pack_samples=args.pack_samples,
            # min_length=args.block_size,
            tokenizer=tokenizer if args.use_subwords else None,
        )

        actual_block_size = args.block_size - 2 if args.use_subwords else args.block_size

        if len(input_ids) > args.block_size:
            start = np.random.randint(0, len(input_ids) - actual_block_size)
            input_ids = input_ids[start : start + actual_block_size]
            labels = labels[start : start + actual_block_size]
        elif len(input_ids) < actual_block_size:
            padding = actual_block_size - len(input_ids)
            # print(padding, lang)
            input_ids += [tokenizer.pad_token_id] * padding if tokenizer else [0] * padding
            labels += [0] * padding

        if tokenizer:
            input_ids = [tokenizer.cls_token_id] + input_ids[:actual_block_size] + [tokenizer.sep_token_id]
            # labels for CLS and SEP tokens are 0 (none)
            labels = [0] + labels[:actual_block_size] + [0]
        else:
            input_ids = input_ids[:actual_block_size]
            labels = labels[:actual_block_size]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        label_weights = torch.ones(args.block_size, dtype=torch.float32)
        if tokenizer:
            attention_mask = (input_ids != tokenizer.pad_token_id).to(torch.float32)
        else:
            attention_mask = (input_ids != 0).to(torch.float32)

        all_input_ids.append(input_ids)
        all_label_weights.append(label_weights)
        all_labels.append(labels)
        all_language_ids.append(Constants.LANG_CODE_TO_INDEX[lang] if add_lang_ids else 0)

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
    if xm.xrt_world_size() == 4:
        # ensure same batch size on TPUv3 and TPUv4
        training_args.per_device_train_batch_size *= 2
    logger.warning(f"Per device train batch size: {training_args.per_device_train_batch_size}")
    logger.warning(
        f"Total train batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps* xm.xrt_world_size()}"
    )

    setup_logging(training_args)
    set_seed(training_args.seed)
    training_args.hub_strategy = "end"
    training_args.save_total_limit = 1

    num_labels = Constants.AUX_OFFSET + ((1 + len(Constants.PUNCTUATION_CHARS)) if args.do_auxiliary_training else 0)
    if args.use_subwords:
        if args.from_scratch:
            config = SubwordXLMConfig(
                args.model_name_or_path,
                num_hidden_layers=args.num_hidden_layers,
                num_labels=num_labels,
                lookahead=args.lookahead,
                lookahead_split_layers=args.lookahead_split_layers,
            )
            backbone = SubwordXLMForTokenClassification(config)

        else:
            config = SubwordXLMConfig.from_pretrained(
                args.model_name_or_path,
                num_hidden_layers=args.num_hidden_layers,
                num_labels=num_labels,
                lookahead=args.lookahead,
                lookahead_split_layers=args.lookahead_split_layers,
            )
            backbone = SubwordXLMForTokenClassification.from_pretrained(
                args.model_name_or_path,
                config=config,
            )

        backbone.config.base_model = args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        if args.lookahead:
            assert args.lookahead % args.num_hidden_layers == 0

    model = Model(
        backbone,
        loss_margin=args.loss_margin,
        use_loss_weights=args.use_loss_weights,
        do_sentence_training=args.do_sentence_training,
        do_auxiliary_training=args.do_auxiliary_training,
        aux_training_weight=args.aux_training_weight,
    )

    if training_args.local_rank == 0:
        logger.warning(summary(model, depth=4))

    def prepare_dataset(
        num_workers=1,
        include_languages=None,
        shuffle=False,
        split="train",
    ):
        with training_args.main_process_first():
            dlconf = DownloadConfig(cache_dir="/home/Markus/.cache/huggingface/datasets")
            dataset = load_dataset("markus583/mC4-TEST", split=split, download_config=dlconf)
        logger.warning(f"Loaded {split} dataset.")
        # optional: delete downloaded dataset, it is stored in cache_dir now (but we delete it later)
        # ~40GB on disk
        # os.system("rm -rf /home/Markus/.cache/huggingface/datasets")

        if include_languages is not None:
            include_languages = set(include_languages)

            dataset = dataset.filter(
                lambda example: example["lang"] in include_languages,
                num_proc=args.preprocessing_num_workers,
            )
            logger.warning(f"Filtered to {len(dataset)} examples.")

        if shuffle:
            dataset = dataset.shuffle(seed=42)
            logger.warning("Shuffled dataset.")

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
                punctuation_indices = {
                    i for i in np.where(examples["ends_with_punctuation"])[0] if i in include_indices
                }

                target_n_non_punct = int(
                    (len(punctuation_indices) * args.non_punctuation_sample_ratio)
                    / (1 - args.non_punctuation_sample_ratio)
                )
                n_drop = (len(include_indices) - len(punctuation_indices)) - target_n_non_punct

                out = [True for _ in range(len(examples["ends_with_punctuation"]))]

                if n_drop <= 0:
                    return out
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

        if args.do_auxiliary_training:
            assert label_args.use_auxiliary

        if args.pack_samples:
            assert not args.one_sample_per_line

        if split == "train" and args.use_subwords:
            with training_args.main_process_first():
                for root, dirs, files in os.walk(os.environ.get("HF_DATASETS_CACHE")):
                    for file in files:
                        if file.startswith("m_c4-test-train"):
                            logger.warning(f"Removing {os.path.join(root, file)}")
                            os.remove(os.path.join(root, file))

        if not args.one_sample_per_line:
            with training_args.main_process_first():
                dataset = dataset.map(
                    pack_sentences,
                    batched=True,
                    num_proc=num_workers,
                    fn_kwargs={
                        "block_size": args.block_size,
                        "tokenizer": tokenizer,
                        "overflow_size": args.overflow_size,
                    },
                    # a bit hacky but oh well, only drop if sentence
                    remove_columns=["ends_with_punctuation", "text"],
                    # load_from_cache_file=False
                )
        logger.warning(f"Grouped {split} dataset.")

        return dataset

    valid_dataset = prepare_dataset(
        num_workers=args.preprocessing_num_workers,
        include_languages=args.include_languages,
        shuffle=False,
        split="valid",
    )
    logger.warning(f"Valid dataset has {len(valid_dataset)} examples.")

    train_dataset = prepare_dataset(
        num_workers=args.preprocessing_num_workers,
        include_languages=args.include_languages,
        shuffle=args.shuffle,
        split="train",
    )
    logger.warning(f"Train dataset has {len(train_dataset)} examples.")

    # print some samples from the dataset
    count = 0
    while count < 5:
        index = random.choice(range(len(train_dataset)))
        sample = train_dataset[index]

        logger.warning(f"Sample {index} of the training set: {sample}.")
        if tokenizer:
            logger.warning(tokenizer.decode(sample["input_ids"]))
        count += 1

    eval_data = torch.load(
        args.eval_data_path,
    )

    def compute_metrics(trainer):
        metrics = {}
        avg_metrics = defaultdict(lambda: [])

        model = trainer._wrap_model(trainer.model, training=False)

        for lang_code, lang_data in tqdm(eval_data.items(), desc="Evaluate!"):
            if args.include_languages is not None and lang_code not in args.include_languages:
                continue

            if trainer.args.process_index == 0 and args.do_sentence_training:
                # with training_args.main_process_first():
                for dataset_name, dataset in lang_data["sentence"].items():
                    # if "corrupt" in dataset_name:
                    #     continue
                    score, info = evaluate_sentence(
                        lang_code,
                        dataset["data"],
                        model,
                        stride=128,
                        block_size=512,
                        batch_size=training_args.per_device_eval_batch_size,
                        threshold=args.threshold,
                    )
                    metrics[f"{lang_code}_{dataset_name}_pr_auc"] = score
                    metrics[f"{lang_code}_{dataset_name}_f1"] = info["f1"]
                    metrics[f"{lang_code}_{dataset_name}_f1_best"] = info["f1_best"]
                    metrics[f"{lang_code}_{dataset_name}_threshold_best"] = info["threshold_best"]
                    avg_metrics[f"average_{dataset_name}_pr_auc"].append(score)
                    avg_metrics[f"average_{dataset_name}_f1"].append(info["f1"])
                    avg_metrics[f"average_{dataset_name}_f1_best"].append(info["f1_best"])
                    avg_metrics[f"average_{dataset_name}_threshold_best"].append(info["threshold_best"])
                    # if lang_code in ["zh", "ja", "my", "km"]:
                    #     avg_metrics[f"average_nonwhitespace_{dataset_name}_pr_auc"].append(score)
                    # else:
                    #     avg_metrics[f"average_whitespace_{dataset_name}_pr_auc"].append(score)
                    # score, _ = evaluate_sentence(
                    #     lang_code,
                    #     dataset["data"],
                    #     model,
                    #     stride=args.eval_stride,
                    #     block_size=args.block_size,
                    #     batch_size=training_args.per_device_eval_batch_size,
                    #     do_lowercase=True,
                    #     do_remove_punct=True,
                    # )
                    # metrics[f"lower_rmp_{lang_code}_{dataset_name}_pr_auc"] = score
                    # avg_metrics[f"lower_rmp_average_{dataset_name}_pr_auc"].append(score)
                    # if lang_code in ["zh", "ja", "my", "km"]:
                    #     avg_metrics[f"lower_rmp_average_nonwhitespace_{dataset_name}_pr_auc"].append(score)
                    # else:
                    #     avg_metrics[f"lower_rmp_average_whitespace_{dataset_name}_pr_auc"].append(score)
                    # k-mer based evaluation
                    # for k in [2, 3, 4]:
                    #     score, avg_acc, info = evaluate_sentence_kmers(
                    #         lang_code,
                    #         dataset["data"],
                    #         model,
                    #         stride=128,
                    #         block_size=512,
                    #         batch_size=training_args.per_device_eval_batch_size,
                    #         k=k,
                    #         # sample_pct=0.1,
                    #         threshold=args.threshold,
                    #     )
                    #     metrics[f"k_{k}_{lang_code}_{dataset_name}_pr_auc"] = score
                    #     avg_metrics[f"k_{k}_average_{dataset_name}_pr_auc"].append(score)
                    #     metrics[f"k_{k}_{lang_code}_{dataset_name}_acc"] = avg_acc
                    #     avg_metrics[f"k_{k}_average_{dataset_name}_acc"].append(avg_acc)
                    #     metrics[f"k_{k}_{lang_code}_{dataset_name}_f1"] = info["f1"]
                    #     metrics[f"k_{k}_{lang_code}_{dataset_name}_f1_best"] = info["f1_best"]
                    #     metrics[f"k_{k}_{lang_code}_{dataset_name}_threshold_best"] = info["threshold_best"]
                    #     avg_metrics[f"k_{k}_average_{dataset_name}_f1"].append(info["f1"])
                    #     avg_metrics[f"k_{k}_average_{dataset_name}_f1_best"].append(info["f1_best"])
                    #     avg_metrics[f"k_{k}_average_{dataset_name}_threshold_best"].append(info["threshold_best"])

                    #     # if lang_code in ["zh", "ja", "my", "km"]:
                    #     #     avg_metrics[f"k_{k}_average_nonwhitespace_{dataset_name}_pr_auc"].append(score)
                    #     #     avg_metrics[f"k_{k}_average_nonwhitespace_{dataset_name}_acc"].append(avg_acc)
                    #     # else:
                    #     #     avg_metrics[f"k_{k}_average_whitespace_{dataset_name}_pr_auc"].append(score)
                    #     #     avg_metrics[f"k_{k}_average_whitespace_{dataset_name}_acc"].append(avg_acc)
                    #     if k == 2:
                    #         # keep keys for backwards compat in wandb
                    #         metrics[f"pairwise_{lang_code}_{dataset_name}_pr_auc"] = score
                    #         avg_metrics[f"pairwise_average_{dataset_name}_pr_auc"].append(score)
                    #         metrics[f"pairwise_{lang_code}_{dataset_name}_acc"] = avg_acc
                    #         avg_metrics[f"pairwise_average_{dataset_name}_acc"].append(avg_acc)
                    #         metrics[f"pairwise_{lang_code}_{dataset_name}_f1"] = info["f1"]
                    #         metrics[f"pairwise_{lang_code}_{dataset_name}_f1_best"] = info["f1_best"]
                    #         metrics[f"pairwise_{lang_code}_{dataset_name}_threshold_best"] = info["threshold_best"]
                    #         avg_metrics[f"pairwise_average_{dataset_name}_f1"].append(info["f1"])
                    #         avg_metrics[f"pairwise_average_{dataset_name}_f1_best"].append(info["f1_best"])
                    #         avg_metrics[f"pairwise_average_{dataset_name}_threshold_best"].append(info["threshold_best"])
                    if lang_code in ["zh", "ja", "my", "km"]:
                        avg_metrics[f"average_nonwhitespace_{dataset_name}_pr_auc"].append(score)
                        avg_metrics[f"average_nonwhitespace_{dataset_name}_f1"].append(info["f1"])
                        avg_metrics[f"average_nonwhitespace_{dataset_name}_f1_best"].append(info["f1_best"])
                        avg_metrics[f"average_nonwhitespace_{dataset_name}_threshold_best"].append(
                            info["threshold_best"]
                        )
                    else:
                        avg_metrics[f"average_whitespace_{dataset_name}_pr_auc"].append(score)
                        avg_metrics[f"average_whitespace_{dataset_name}_f1"].append(info["f1"])
                        avg_metrics[f"average_whitespace_{dataset_name}_f1_best"].append(info["f1_best"])
                        avg_metrics[f"average_whitespace_{dataset_name}_threshold_best"].append(info["threshold_best"])

        for name, values in avg_metrics.items():
            if len(values) > 1:
                metrics[name] = np.mean(values)

        return metrics

    if "wandb" in training_args.report_to and training_args.process_index == 0:
        wandb.init(name=wandb_name, project="sentence", entity="markus_583")
        wandb.config.update(args)
        wandb.config.update(training_args)
        wandb.config.update(label_args)

        model.config.wandb_run_id = wandb.run.id

        for file in glob(os.path.join(os.path.dirname(__file__), "*.py")):
            wandb.save(os.path.abspath(file), policy="now")
            # also 1 dir above
            wandb.save(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", file)), policy="now")

    label_dict = get_subword_label_dict(label_args, tokenizer) if args.use_subwords else get_label_dict(label_args)
    logger.info(f"Label dict has {len(label_dict)} entries.")

    # needed in the trainer
    training_args.adapter_warmup_steps = args.adapter_warmup_steps
    training_args.adapter_lr_multiplier = args.adapter_lr_multiplier

    # give .map in multiprocessing enough of time to finish, to be safe
    time.sleep(10)
    if training_args.local_rank == 0:
        # since both share the *same* cache_dir, we cannot simply call dataset.cleanup_cache_files()
        # because that would remove the cache files of the other dataset!
        cleanup_cache_files([train_dataset, valid_dataset])
        logger.warning("Cleaned up cache files.")
    time.sleep(10)

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
            add_lang_ids=not args.use_subwords,
        ),
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()
    # Pattern for checkpoint directories
    checkpoint_pattern = os.path.join(training_args.output_dir, "checkpoint-*")

    # Use glob.glob to find all directories matching the pattern
    for checkpoint_dir in glob(checkpoint_pattern):
        if os.path.isdir(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    # try:
    main()
    # except Exception:
    #     # extype, value, tb = sys.exc_info()
    #     # tb.print_exc()
    #     # pdb.post_mortem(tb)
    #     pass
