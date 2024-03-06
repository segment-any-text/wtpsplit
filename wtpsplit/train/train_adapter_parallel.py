import copy
import dataclasses
import json
import logging
import math
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from glob import glob
from typing import List

import datasets
import numpy as np
import torch
import torch_xla.core.xla_model as xm
import transformers
from tokenizers import AddedToken
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, set_seed

import adapters
import wandb
from adapters import AdapterArguments
from wtpsplit.models import SubwordXLMConfig, SubwordXLMForTokenClassification
from wtpsplit.train.adapter_utils import (
    ParallelTPUAdapterTrainingArguments as TrainingArguments,
)
from wtpsplit.train.adapter_utils import (
    ParallelTPUWandbCallback as WandbCallback,
)
from wtpsplit.train.adaptertrainer import AdapterTrainer
from wtpsplit.train.evaluate import evaluate_sentence, evaluate_sentence_pairwise
from wtpsplit.train.train import collate_fn
from wtpsplit.train.utils import Model
from wtpsplit.utils import Constants, LabelArgs, get_label_dict, get_subword_label_dict
from wtpsplit.evaluation.intrinsic import corrupt

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_logging(training_args, job_id=None) -> None:
    # Generate a unique logger name based on the job_id or process identifier
    unique_logger_name = f"{__name__}.{job_id}" if job_id is not None else __name__
    logger = logging.getLogger(unique_logger_name)

    # Clear existing handlers to avoid duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    # Disable propagation to prevent logs from being handled elsewhere
    logger.propagate = False  # Note the correct attribute is `propagate`

    # Set the logger's level based on training arguments
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    # Create and add a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(console_handler)

    # Add a file handler if a job_id is provided, open in write mode to start from scratch
    if job_id is not None:
        file_handler = logging.FileHandler(f"logs/log_{job_id}.log", mode="w")  # Open in write mode
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
        logger.addHandler(file_handler)

    # Adjust verbosity settings for datasets and transformers
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log a summary message using the newly configured logger
    logger.warning(
        (
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            + f"distributed training: {training_args.local_rank != -1}, 16-bits training: {training_args.fp16}"
        )
    )

    # Return the configured logger for use in the rest of the process
    return logger


@dataclass
class Args:
    model_name_or_path: str
    base_model: str = "xlm-roberta-base"
    shuffle: bool = True
    text_path: str = "data/eval.pth"
    include_languages: List[str] = None
    preprocessing_num_workers: int = 1
    block_size: int = 512
    overflow_size: int = 16
    eval_stride: int = 256
    loss_margin: float = 0.5
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
    freeze_classifier: bool = False
    clf_from_scratch: bool = False
    unfreeze_ln: bool = False
    do_process: bool = False
    n_train_steps: List[int] = field(default_factory=lambda: [1000, 10000, 100000])
    meta_clf: bool = False
    wandb_project = "sentence"
    # corruption
    do_lowercase: bool = False
    do_remove_punct: bool = False
    eval_pairwise: bool = False


def main(
    tpu_core_idx,
    args,
    training_args,
    label_args,
    adapter_args,
    data,
    train_ds,
    valid_ds,
    lang_groups,
    train_steps,
):
    wandb_name = training_args.output_dir

    logger = setup_logging(training_args, job_id=tpu_core_idx)
    set_seed(training_args.seed)
    logger.warning(f"{tpu_core_idx}: LANG GROUP {lang_groups}")

    num_labels = Constants.AUX_OFFSET + (
        (1 + len(Constants.PUNCTUATION_CHARS)) if (label_args.use_auxiliary or args.do_auxiliary_training or args.meta_clf) else 0
    )
    config = SubwordXLMConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
    )

    # 1 wandb run for all language-dataset combinations
    if "wandb" in training_args.report_to:
        wandb.init(name=f"{wandb_name}-{tpu_core_idx}", project=args.wandb_project, group=wandb_name)
        wandb.config.update(args)
        wandb.config.update(training_args)
        wandb.config.update(label_args)
        wandb.config.update(adapter_args)

        for file in glob(os.path.join(os.path.dirname(__file__), "*.py")):
            wandb.save(os.path.abspath(file), policy="now")
        wandb.log({"train/total_n_batches": len(lang_groups)})
        training_args.report_to = []
        callbacks = WandbCallback()
    else:
        callbacks = None

    xm.rendezvous("wandb init done")

    for i, ((lang, dataset_name), train_step) in tqdm(enumerate(zip(lang_groups, train_steps)), total=len(lang_groups)):
        # do model stuff here; otherwise, head params would be overwritten every time
        backbone = SubwordXLMForTokenClassification.from_pretrained(
            args.model_name_or_path, config=copy.deepcopy(config), ignore_mismatched_sizes=True
        )
        logger.warning(f"{tpu_core_idx}: Loaded backbone {args.model_name_or_path}.")
        backbone.config.base_model = args.base_model

        # setup adapters
        model_type = backbone.config.model_type
        # adapters need xlm-roberta as model type.
        backbone.config.model_type = "xlm-roberta"  # needed for adapter setup
        adapters.init(backbone)
        # reset model type (used later)
        backbone.config.model_type = model_type

        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        # needed since we create labels in collate_fn based on tokens
        tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})

        model = Model(
            backbone,
            loss_margin=args.loss_margin,
            use_loss_weights=args.use_loss_weights,
            do_sentence_training=args.do_sentence_training,
            do_auxiliary_training=args.do_auxiliary_training,
            aux_training_weight=args.aux_training_weight,
        )

        # train for as many steps as the current group's steps.
        training_args.max_steps = train_step
        training_args.evaluation_strategy = "steps"
        training_args.eval_steps = (train_step // training_args.num_train_epochs) + 1

        # print some samples from the dataset
        count = 0
        while count < 0:
            index = random.choice(range(len(train_ds[(lang, dataset_name)])))
            sample = train_ds[(lang, dataset_name)][index]

            logger.warning(f"{tpu_core_idx}: Sample {index} of the training set: {sample}.")
            if tokenizer:
                logger.warning(tokenizer.decode(sample["input_ids"]))
            count += 1

        def compute_metrics(trainer):
            metrics = {}
            eval_data = data[lang]["sentence"][dataset_name]["data"]

            model = trainer._wrap_model(trainer.model, training=False)

            score, info = evaluate_sentence(
                lang,
                eval_data,
                model,
                stride=64,
                block_size=512,
                batch_size=training_args.per_device_eval_batch_size,
            )
            metrics[f"{dataset_name}/{lang}/pr_auc"] = score
            metrics[f"{dataset_name}/{lang}/f1"] = info["f1"]
            metrics[f"{dataset_name}/{lang}/f1_best"] = info["f1_best"]
            metrics[f"{dataset_name}/{lang}/threshold_best"] = info["threshold_best"]
            if args.do_lowercase and args.do_remove_punct: 
                score_corrupted, info_corrupted = evaluate_sentence(
                    lang,
                    eval_data,
                    model,
                    stride=64,
                    block_size=512,
                    batch_size=training_args.per_device_eval_batch_size,
                    do_lowercase=True,
                    do_remove_punct=True    
                )
                metrics[f"{dataset_name}/{lang}/corrupted/pr_auc"] = score_corrupted
                metrics[f"{dataset_name}/{lang}/corrupted/f1"] = info_corrupted["f1"]
                metrics[f"{dataset_name}/{lang}/corrupted/f1_best"] = info_corrupted["f1_best"]
                metrics[f"{dataset_name}/{lang}/corrupted/threshold_best"] = info_corrupted["threshold_best"]
            elif args.do_lowercase or args.do_remove_punct:
                raise NotImplementedError("Currently we only corrupt both ways!")
            if args.eval_pairwise:
                score_pairwise, avg_acc = evaluate_sentence_pairwise(
                    lang,
                    eval_data,
                    model,
                    stride=args.eval_stride,
                    block_size=args.block_size,
                    batch_size=training_args.per_device_eval_batch_size,
                    threshold=0.1,
                )
                metrics[f"{dataset_name}/{lang}/pairwise/pr_auc"] = score_pairwise
                metrics[f"{dataset_name}/{lang}/pairwise/acc"] = avg_acc
            xm.rendezvous("eval log done")

            return metrics

        label_dict = get_subword_label_dict(label_args, tokenizer) if args.use_subwords else get_label_dict(label_args)

        # init new adapter
        model.backbone.add_adapter("text", config=adapter_args.adapter_config, set_active=True, overwrite_ok=True)
        model.backbone.train_adapter("text")
        if tpu_core_idx == 0:
            logger.warning(f"{tpu_core_idx}: {model.backbone.adapter_summary()}")

        if args.freeze_classifier:
            for n, p in model.backbone.named_parameters():
                if "classifier" in n:
                    p.requires_grad = False
        if args.clf_from_scratch:
            model.backbone.classifier = torch.nn.Linear(model.backbone.config.hidden_size, num_labels)
            
        if args.unfreeze_ln:
            for n, p in model.backbone.named_parameters():
                if "LayerNorm" in n:
                    p.requires_grad = True    
        
        if args.meta_clf:
            clf = model.backbone.classifier
            model.backbone.classifier = torch.nn.Sequential(
                clf,  # original classifier - if frozen above, also frozen here
                torch.nn.Linear(clf.out_features, 1)
            )
            model.backbone.config.num_labels = 1

        trainer = AdapterTrainer(
            model,
            training_args,
            train_dataset=train_ds[(lang, dataset_name)],
            eval_dataset=valid_ds[(lang, dataset_name)],
            compute_metrics=compute_metrics,
            data_collator=partial(
                collate_fn,
                args=args,
                label_args=label_args,
                label_dict=label_dict,
                tokenizer=tokenizer,
            ),
            logging_prefix=f"{dataset_name}/{lang}/",
        )
        if callbacks:
            trainer.add_callback(callbacks)

        logger.warning(f"{tpu_core_idx}: START TRAIN {lang} {dataset_name}.")
        # wait until all TPUs are ready
        xm.rendezvous("start_training")

        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

        if not os.path.exists(os.path.join(training_args.output_dir, dataset_name, lang)):
            os.makedirs(os.path.join(training_args.output_dir, dataset_name, lang))
        save_model = copy.deepcopy(model.backbone)
        save_model = save_model.to("cpu")
        save_model.save_adapter(
            adapter_name="text",
            save_directory=os.path.join(training_args.output_dir, dataset_name, lang),
            with_head=True,
        )

        if args.unfreeze_ln:
            # no way within adapters to do this, need to do it manually
            ln_dict = {n: p for n, p in save_model.named_parameters() if "LayerNorm" in n}
            torch.save(ln_dict, os.path.join(training_args.output_dir, dataset_name, lang, "ln_dict.pth"))
        logger.warning(f"{tpu_core_idx}: DONE TRAIN {lang} {dataset_name}.")

        if callbacks:
            wandb.log({"train/batch_progress": (i + 1) / len(lang_groups)})

        xm.rendezvous("end_training")
        xm.mark_step()
    xm.rendezvous("all_done")
    wandb.finish()


# split languages into groups of equal size for TPUs
def split_langs_into_groups(langs, n_groups=8):
    return [langs[i::n_groups] for i in range(n_groups)]


def _mp_fn(index):
    # For xla_spawn (TPUs)
    setup(index)


def setup(index):
    config_path = sys.argv[1]
    parser = HfArgumentParser([Args, TrainingArguments, LabelArgs, AdapterArguments])
    (args, training_args, label_args, adapter_args) = parser.parse_json_file(config_path)

    data = torch.load(
        args.text_path,
    )
    if index == 0:
        print(f"Using {xm.xrt_world_size()} processes/TPUs.")
        print("Loaded data.")
        print(f"Using step sizes {args.n_train_steps}.")
        # create a csv file that writes the length of each train dataset
        # used to sort the datasets by length and assign them to workers
        if not os.path.exists("logs"):
            os.makedirs("logs", exist_ok=True)
        with open("logs/train_dataset_lengths.csv", "w") as f:
            f.write("lang,dataset_name,token_length,original_length\n")

    def prepare_dataset(
        data,
        num_workers=1,
        include_languages=None,
        dataset_name="ud",
        shuffle=False,
        split="train",
        do_lowercase=False,
        do_remove_punct=False,
    ):
        # maybe we use more than 1 lang later at once.
        for lang in include_languages:
            if split == "train":
                dataset = data[lang]["sentence"][dataset_name]["meta"]["train_data"]
            elif split == "valid":
                dataset = data[lang]["sentence"][dataset_name]["data"]
            if dataset is None:
                return None
            dataset = datasets.Dataset.from_list(
                [
                    {
                        args.text_column: corrupt(sample, do_lowercase, do_remove_punct) + "\n" if sample and sample[-1] != "\n" else corrupt(sample, do_lowercase, do_remove_punct),
                        "lang": lang,
                        "ends_with_punctuation": sample.endswith(tuple(Constants.PUNCTUATION_CHARS)),
                    }
                    for sample in dataset
                ]
            )

        if shuffle:
            dataset = dataset.shuffle(seed=42)

        # very likely not relevant / used only for the compound part
        if args.ignore_non_hyphen:
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

            for current_lang in set(examples["lang"]):
                # only retain current_lang examples (all columns)
                lang_subwords = [
                    subwords for subwords, lang in zip(examples["input_ids"], examples["lang"]) if lang == current_lang
                ]
                # filter out some special tokens
                # from html tags, mostly in Latin, Thai & Korean
                lang_subwords = [
                    [subword for subword in subwords if subword not in special_tokens_ids] for subwords in lang_subwords
                ]
                # concatenate token lists
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

        if args.pack_samples:
            raise NotImplementedError("Packing samples not implemented for subword-based models.")

        if args.use_subwords:
            dataset = dataset.map(
                tokenize_texts,
                batched=True,
                num_proc=num_workers,
                remove_columns=[args.text_column],
            )
        else:
            # this is no longer used and would cause an error otherwise
            dataset = dataset.rename_column(args.text_column, "input_ids")

        if not args.one_sample_per_line:
            dataset = dataset.map(
                group_texts,
                batched=True,
                num_proc=num_workers,
                # a bit hacky but oh well, only drop if sentence
                remove_columns=["ends_with_punctuation"] if args.text_column == "text" else [],
            )

        return dataset

    if not args.include_languages:
        args.include_languages = list(data.keys())  # use all
        # XXX: for testing
        # args.include_languages = ["af", "az", "kk", "te", "tg", "be", "km",]  # "ps", "ru"]
    # get all lang-dataset combinations and their lengths
    all_keys = []
    for lang in data.keys():
        for dataset_name in data[lang]["sentence"].keys():
            if lang in args.include_languages:
                valid = data[lang]["sentence"][dataset_name]["data"]
                train = data[lang]["sentence"][dataset_name]["meta"]["train_data"]
                if train is not None and valid is not None:
                    all_keys.append((lang, dataset_name, len(train)))
    # sort by length of train dataset
    all_keys = sorted(all_keys, key=lambda x: x[2], reverse=True)
    all_lang_groups = split_langs_into_groups(list(all_keys), n_groups=int(xm.xrt_world_size()))
    current_lang_groups = [(lang, ds) for (lang, ds, _) in all_lang_groups[index]]
    # TODO: check speed of parallelism here
    # should be: longest (parallel), ..., shortest (parallel)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # needed since we create labels in collate_fn based on tokens
    tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})
    custom_token_id = tokenizer.convert_tokens_to_ids("\n")
    # used later to filter out special tokens
    special_tokens_ids = set(tokenizer.all_special_ids)
    special_tokens_ids.discard(custom_token_id)

    xm.rendezvous("loading data")

    if not os.path.exists("data/ft_data"):
        os.makedirs("data/ft_data", exist_ok=True)

    def process_datasets(data, args, current_lang_groups, do_process=True, do_write=True):
        all_ds = {"train": {}, "valid": {}}

        for lang in data.keys():
            if lang in args.include_languages:
                for dataset_name in data[lang]["sentence"].keys():
                    if (lang, dataset_name) not in current_lang_groups:
                        continue
                    train_path = f"data/ft_data/{lang}_{dataset_name}_train.pth"
                    valid_path = f"data/ft_data/{lang}_{dataset_name}_valid.pth"

                    if not do_process and os.path.exists(train_path) and os.path.exists(valid_path):
                        # if exists and we don't want to process, load
                        train_dataset = torch.load(train_path)
                        valid_dataset = torch.load(valid_path)

                        all_ds["train"][(lang, dataset_name)] = train_dataset
                        all_ds["valid"][(lang, dataset_name)] = valid_dataset
                    else:
                        # Process datasets
                        valid_dataset = prepare_dataset(
                            data=data,
                            num_workers=1,
                            include_languages=[lang],
                            dataset_name=dataset_name,
                            shuffle=False,
                            split="valid",
                            do_lowercase=args.do_lowercase,
                            do_remove_punct=args.do_remove_punct,
                        )

                        train_dataset = prepare_dataset(
                            data=data,
                            num_workers=args.preprocessing_num_workers,
                            include_languages=[lang],
                            dataset_name=dataset_name,
                            shuffle=args.shuffle,
                            split="train",
                            do_lowercase=args.do_lowercase,
                            do_remove_punct=args.do_remove_punct,
                        )

                        all_ds["valid"][(lang, dataset_name)] = valid_dataset
                        all_ds["train"][(lang, dataset_name)] = train_dataset

                        torch.save(train_dataset, train_path)
                        torch.save(valid_dataset, valid_path)

                    # Write length of train dataset to CSV
                    if do_write:
                        print(f"Valid ds for {lang} {dataset_name} has {len(valid_dataset)} examples.")
                        print(f"Train ds for {lang} {dataset_name} has {len(train_dataset)} examples.")
                        with open("logs/train_dataset_lengths.csv", "a") as f:
                            train_data_len = len(data[lang]["sentence"][dataset_name]["meta"]["train_data"])
                            f.write(f"{lang},{dataset_name},{len(train_dataset)},{train_data_len}\n")

        if do_process and do_write and index == 0:
            with open("data/ft_data/args.json", "w") as f:
                json.dump(dataclasses.asdict(args), f)
        return all_ds

    # first, pre-process datasets in distributed manner
    # assignment of lang-dataset combinations to workers is done based on length of train dataset (string length)
    _ = process_datasets(data, args, current_lang_groups, do_process=args.do_process, do_write=True)
    xm.rendezvous("data loaded")

    # synchronize number of steps before training, for each worker
    with open("logs/train_dataset_lengths.csv", "r") as f:
        lines = f.readlines()
        lines = [line.strip().split(",") for line in lines[1:]]
        lines = sorted(lines, key=lambda x: int(x[2]), reverse=True)
        # as tuple
        lines = [
            (
                x[0],
                x[1],
                int(x[2]),
                int(x[3]),
                # calculate number of steps based on train dataset token length
                # XXX: steps are dependent on epoch, too! So we set target number of epochs and calculate steps based on that
                math.ceil(
                    (training_args.num_train_epochs * int(x[2]))
                    / (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
                ),
            )
            for x in lines
        ]

    all_keys = lines  # now contains lang, dataset, token length (!), original length, original train steps

    # bin the keys into groups based on number of steps
    grouped_keys = {n_steps: [x for x in all_keys if x[4] <= n_steps] for n_steps in args.n_train_steps[:-1]}
    # last group is all keys with >10_000 steps
    grouped_keys[args.n_train_steps[-1]] = [x for x in all_keys if x[4] > args.n_train_steps[-2]]

    # split each group into equal parts for each worker
    grouped_lang_groups = {
        n_steps: split_langs_into_groups(grouped_keys[n_steps], n_groups=xm.xrt_world_size())
        for n_steps in args.n_train_steps
    }
    # ensure each last group is of equal length
    for n_steps in args.n_train_steps:
        for i, group in enumerate(grouped_lang_groups[n_steps]):
            if len(group) < len(grouped_lang_groups[n_steps][0]):
                grouped_lang_groups[n_steps][i].append(grouped_lang_groups[n_steps][0][-1])
        assert all([len(group) == len(grouped_lang_groups[n_steps][0]) for group in grouped_lang_groups[n_steps]])

    # unwrap dict of lists (just remove dict dimension)
    all_lang_groups = []

    def merge_dict_into_sublists(d):
        # Initialize a list with 8 empty sublists
        merged_lists = [[] for _ in range(xm.xrt_world_size())]

        # Sort keys in descending order and iterate through them
        for key in sorted(d.keys(), reverse=True):
            # Iterate through each of the 8 sublists for the current key
            for index, sublist in enumerate(d[key]):
                # add key (number of used steps) to each item in the sublist
                merged_lists[index].extend([item + (key,) for item in sublist])
        return merged_lists

    all_lang_groups = merge_dict_into_sublists(grouped_lang_groups)
    train_steps = [x[5] for x in all_lang_groups[index]]
    current_lang_groups = [(x[0], x[1]) for x in all_lang_groups[index]]

    all_ds = process_datasets(data, args, current_lang_groups, do_process=False, do_write=False)

    # all lang groups should be of equal length
    assert all([len(lang_group) == len(all_lang_groups[0]) for lang_group in all_lang_groups])
    # all lang groups should contain unique lang-dataset combinations
    assert all([len(lang_group) == len(set(lang_group)) for lang_group in all_lang_groups])

    if index == 0:
        # just for sanity chacking
        with open("logs/all_lang_groups.txt", "w") as f:
            f.write(f"{training_args.num_train_epochs}\n")
            f.write("\n".join([str(x) for x in all_lang_groups]))
            print(all_lang_groups)
    xm.rendezvous("data sorted")

    main(
        index,
        args,
        training_args,
        label_args,
        adapter_args,
        data,
        all_ds["train"],
        all_ds["valid"],
        current_lang_groups,
        train_steps,
    )
    
    xm.rendezvous("all training done")
    if index == 0:
        # eval here within 1 go
        if args.do_lowercase and args.do_remove_punct:
            os.system(
                f"python3 wtpsplit/evaluation/intrinsic.py --model_path {args.model_name_or_path} --adapter_path {training_args.output_dir} --threshold 0.1 --do_lowercase --do_remove_punct"
            )
        elif args.eval_pairwise:
            os.system(
                f"python3 wtpsplit/evaluation/intrinsic_pairwise.py --model_path {args.model_name_or_path} --adapter_path {training_args.output_dir} --threshold 0.1"
           )
        elif "lines" in args.text_path:
            os.system(
                f"python3 wtpsplit/evaluation/intrinsic.py --model_path {args.model_name_or_path} --adapter_path {training_args.output_dir} --threshold 0.1--custom_language_list data/lyrics_langs.csv --eval_data_path data/lyrics_lines.pt --save_suffix lines"
            )
        elif "verses" in args.text_path:
            os.system(
                f"python3 wtpsplit/evaluation/intrinsic.py --model_path {args.model_name_or_path} --adapter_path {training_args.output_dir} --threshold 0.1 --custom_language_list data/lyrics_langs.csv --eval_data_path data/lyrics_verses_strip_n.pt --save_suffix verses"
           )
        else:
            os.system(
                f"python3 wtpsplit/evaluation/intrinsic.py --model_path {args.model_name_or_path} --adapter_path {training_args.output_dir} --threshold 0.1"
            )


if __name__ == "__main__":
    import torch_xla.distributed.xla_multiprocessing as xmp

    xmp.spawn(
        _mp_fn,
        args=(),
        nprocs=8,
    )
    