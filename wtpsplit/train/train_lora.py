import copy
import logging
import math
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from functools import partial
from glob import glob
from typing import List, Optional, Union

import adapters
import datasets
import numpy as np
import torch
from adapters import AdapterArguments
from adapters.models import MODEL_MIXIN_MAPPING
from adapters.models.bert.mixin_bert import BertModelAdaptersMixin
from tokenizers import AddedToken
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed

import wandb
from wtpsplit.models import SubwordXLMConfig, SubwordXLMForTokenClassification
from wtpsplit.train.adaptertrainer import AdapterTrainer
from wtpsplit.train.evaluate import evaluate_sentence
from wtpsplit.train.train import collate_fn, setup_logging
from wtpsplit.train.trainer import Trainer
from wtpsplit.train.utils import Model
from wtpsplit.utils import Constants, LabelArgs, get_label_dict, get_subword_label_dict

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_MIXIN_MAPPING["SubwordXLMRobertaModel"] = BertModelAdaptersMixin


@dataclass
class Args:
    model_name_or_path: str
    base_model: str = "xlm-roberta-base"
    shuffle: bool = True
    text_path: str = "data/all_data.pth"
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
    do_auxiliary_training: bool = False
    aux_training_weight: float = 1.0
    ignore_non_hyphen: bool = False
    non_punctuation_sample_ratio: float = None
    adapter_warmup_steps: int = 0
    adapter_lr_multiplier: float = 1.0
    text_column: str = "text"
    num_hidden_layers: int = 0

    # NEW PARAMS
    use_subwords: bool = False
    freeze_classifier: bool = False
    clf_from_scratch: bool = False
    unfreeze_ln: bool = False
    do_process: bool = False
    meta_clf: bool = False
    wandb_project: str = "sentence"
    eval_every: int = 5
    # corruption
    skip_eval_loss: bool = False
    subsample: Optional[float] = None


def main():
    parser = HfArgumentParser([Args, TrainingArguments, LabelArgs, AdapterArguments])
    if sys.argv[1].endswith(".json"):
        (args, training_args, label_args, adapter_args) = parser.parse_json_file(sys.argv[1])
        wandb_name = training_args.output_dir
    else:
        (args, training_args, label_args, adapter_args) = parser.parse_args_into_dataclasses()
        wandb_name = None

    setup_logging(training_args)
    set_seed(training_args.seed)

    num_labels = Constants.AUX_OFFSET + (
        (1 + len(Constants.PUNCTUATION_CHARS))
        if (label_args.use_auxiliary or args.do_auxiliary_training or args.meta_clf)
        else 0
    )

    if args.num_hidden_layers:
        config = SubwordXLMConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            num_hidden_layers=args.num_hidden_layers,
        )
    else:
        config = SubwordXLMConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)

    def prepare_dataset(
        data,
        num_workers=1,
        include_languages=None,
        dataset_name="ud",
        shuffle=False,
        split="train",
        subsample: Union[None, int, float] = None,
        one_sample_per_line: bool = False,
    ):
        with training_args.main_process_first():
            # this only uses 1 dataset-language combination, but can be easily adapted if needed
            for lang in include_languages:
                if split == "train":
                    dataset = data[lang]["sentence"][dataset_name]["meta"]["train_data"]
                elif split == "valid":
                    dataset = data[lang]["sentence"][dataset_name]["data"]
                if dataset is None:
                    return None

                if one_sample_per_line or isinstance(dataset[0], list):
                    processed_dataset = []
                    for chunk in dataset:
                        if "\n" in chunk:
                            raise ValueError(
                                "Newlines in text are not supported! Data needs to be processed as a list of sentences."
                            )
                        processed_chunk = {}
                        processed_chunk["lang"] = lang
                        processed_chunk["ends_with_punctuation"] = chunk[-1].endswith(
                            tuple(Constants.PUNCTUATION_CHARS)
                        )
                        # join all chunks
                        processed_chunk[args.text_column] = "\n".join(chunk)
                        processed_dataset.append(processed_chunk)
                    dataset = datasets.Dataset.from_list(processed_dataset)

                else:
                    for i, chunk in enumerate(dataset):
                        if "\n" in chunk:
                            raise ValueError(
                                "Newlines in text are not supported! Data needs to be processed as a list of sentences."
                            )
                    dataset = datasets.Dataset.from_list(
                        [
                            {
                                args.text_column: sample + "\n" if sample and sample[-1] != "\n" else sample,
                                "lang": lang,
                                "ends_with_punctuation": sample.endswith(tuple(Constants.PUNCTUATION_CHARS)),
                            }
                            for sample in dataset
                        ]
                    )
            with training_args.main_process_first():
                logger.warning(f"Loaded {len(dataset)} examples for {lang} {dataset_name} {split} dataset.")

        if shuffle:
            dataset = dataset.shuffle(seed=training_args.seed)
        if subsample:
            old_length = len(dataset)
            if isinstance(subsample, int):
                # ensure that we don't try to select more than the dataset length
                subsample = min(subsample, len(dataset))
                dataset = dataset.select(range(subsample))
            elif isinstance(subsample, float):
                dataset = dataset.select(range(int(subsample * len(dataset))))
            logger.warning(f"Subsampled {len(dataset)} examples from {old_length}.")

        # ignore
        if args.ignore_non_hyphen:
            with training_args.main_process_first():
                dataset = dataset.filter(
                    lambda sample: any(c in sample[args.text_column] for c in label_args.hyphen_chars),
                    num_proc=args.preprocessing_num_workers,
                )
                with training_args.main_process_first():
                    logger.info(f"Filtered to {len(dataset)} examples.")

        # "punctuation-specific sampling" in the WtP paper
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
                    padding = model.backbone.config.downsampling_rate - (len(text) % model.backbone.downsampling_rate)
                    if padding == model.backbone.downsampling_rate:
                        padding = 0

                    text += chr(0) * padding

                return text

            for current_lang in set(examples["lang"]):
                if not args.use_subwords:
                    lang_texts = [
                        maybe_pad(text)
                        for text, lang in zip(examples["input_ids"], examples["lang"])
                        if lang == current_lang
                    ]
                else:
                    # only retain current_lang examples (all columns)
                    lang_subwords = [
                        subwords
                        for subwords, lang in zip(examples["input_ids"], examples["lang"])
                        if lang == current_lang
                    ]
                    # filter out some special tokens
                    # from html tags, mostly in Latin, Thai & Korean
                    lang_subwords = [
                        [subword for subword in subwords if subword not in special_tokens_ids]
                        for subwords in lang_subwords
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
                        # concatenate token lists
                        concatenated_texts = [item for sublist in lang_subwords for item in sublist]
                        concatenated_ids = [i for i, subwords in enumerate(lang_subwords) for _ in subwords]

                    total_length = len(concatenated_texts)

                    best_length = math.ceil(total_length / args.block_size) * args.block_size + args.overflow_size
                    while best_length > total_length:
                        best_length -= args.block_size
                    if best_length < args.block_size:
                        best_length = args.block_size + 1

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
            assert not one_sample_per_line

        if args.use_subwords:
            with training_args.main_process_first():
                dataset = dataset.map(
                    tokenize_texts,
                    batched=True,
                    num_proc=num_workers,
                    remove_columns=[args.text_column],
                    desc="Tokenizing",
                )
        else:
            # this is no longer used and would cause an error otherwise
            with training_args.main_process_first():
                dataset = dataset.rename_column(args.text_column, "input_ids")

        if not one_sample_per_line:
            with training_args.main_process_first():
                dataset = dataset.map(
                    group_texts,
                    batched=True,
                    num_proc=num_workers,
                    # a bit hacky but oh well, only drop if sentence
                    remove_columns=["ends_with_punctuation"] if args.text_column == "text" else [],
                    desc="Grouping",
                )

        return dataset

    with training_args.main_process_first():
        data = torch.load(
            args.text_path,
        )
        # sort alphabetically by key for alphabetical filtering
        data = dict(sorted(data.items()))

    if not args.include_languages:
        args.include_languages = list(data.keys())  # use all

    # 1 wandb run for all language-dataset combinations
    if "wandb" in training_args.report_to and training_args.process_index == 0:
        wandb.init(name=wandb_name, project=args.wandb_project, group=wandb_name)
        wandb.config.update(args)
        wandb.config.update(training_args)
        wandb.config.update(label_args)
        wandb.config.update(adapter_args)

        for file in glob(os.path.join(os.path.dirname(__file__), "*.py")):
            wandb.save(os.path.abspath(file), policy="now")

    for lang in tqdm(data.keys(), desc="Language"):
        if lang in args.include_languages:
            for dataset_name in data[lang]["sentence"].keys():
                print("RUNNING:", dataset_name, lang)
                # do model stuff here; otherwise, head params would be overwritten every time
                backbone = SubwordXLMForTokenClassification.from_pretrained(
                    args.model_name_or_path, config=copy.deepcopy(config), ignore_mismatched_sizes=True
                )
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
                custom_token_id = tokenizer.convert_tokens_to_ids("\n")
                # used later to filter out special tokens
                special_tokens_ids = set(tokenizer.all_special_ids)
                special_tokens_ids.discard(custom_token_id)

                if "short" in dataset_name:
                    one_sample_per_line = True
                else:
                    one_sample_per_line = args.one_sample_per_line  # used for lyrics

                model = Model(
                    backbone,
                    loss_margin=args.loss_margin,
                    use_loss_weights=args.use_loss_weights,
                    do_sentence_training=args.do_sentence_training,
                    do_auxiliary_training=args.do_auxiliary_training,
                    aux_training_weight=args.aux_training_weight,
                )

                with training_args.main_process_first():
                    valid_dataset = prepare_dataset(
                        data=data,
                        num_workers=1,
                        include_languages=[lang],
                        dataset_name=dataset_name,
                        shuffle=False,
                        split="valid",
                        one_sample_per_line=one_sample_per_line,
                    )
                    logger.warning(f"Valid ds for {lang} {dataset_name} has {len(valid_dataset)} examples.")

                    train_dataset = prepare_dataset(
                        data=data,
                        num_workers=args.preprocessing_num_workers,
                        include_languages=[lang],
                        dataset_name=dataset_name,
                        shuffle=args.shuffle,
                        split="train",
                        subsample=args.subsample,
                        one_sample_per_line=one_sample_per_line,
                    )
                    if train_dataset is None or valid_dataset is None:
                        logger.warning(f"Skipping {lang} {dataset_name} due to missing data.")
                        continue
                    logger.warning(f"Train ds for {lang} {dataset_name} has {len(train_dataset)} examples.")

                # print some samples from the dataset
                count = 0
                while count < 1:
                    index = random.choice(range(len(train_dataset)))
                    sample = train_dataset[index]

                    logger.warning(f"Sample {index} of the training set: {sample}.")
                    if tokenizer:
                        logger.warning(tokenizer.decode(sample["input_ids"]))
                    count += 1

                def compute_metrics(trainer):
                    metrics = {}
                    eval_data = data[lang]["sentence"][dataset_name]["data"]

                    model = trainer._wrap_model(trainer.model, training=False)

                    with training_args.main_process_first():
                        # feeding in single samples is too slow --> feed in as one long text
                        if args.one_sample_per_line:
                            eval_data = [item for sublist in eval_data for item in sublist]
                        elif isinstance(eval_data[0], list):
                            eval_data = [item for sublist in eval_data for item in sublist]
                        score, info = evaluate_sentence(
                            lang,
                            eval_data,
                            model,
                            stride=args.eval_stride,
                            block_size=args.block_size,
                            batch_size=training_args.per_device_eval_batch_size,
                        )
                        metrics[f"{dataset_name}/{lang}/pr_auc"] = score
                        metrics[f"{dataset_name}/{lang}/f1"] = info["f1"]
                        metrics[f"{dataset_name}/{lang}/f1_best"] = info["f1_best"]
                        metrics[f"{dataset_name}/{lang}/threshold_best"] = info["threshold_best"]

                    return metrics

                label_dict = (
                    get_subword_label_dict(label_args, tokenizer) if args.use_subwords else get_label_dict(label_args)
                )

                if adapter_args.train_adapter:
                    # init new adapter
                    model.backbone.add_adapter(
                        "text", config=adapter_args.adapter_config, set_active=True, overwrite_ok=True
                    )
                    model.backbone.train_adapter("text")
                    kwargs = {"logging_prefix": f"{dataset_name}/{lang}/", "skip_eval_loss": args.skip_eval_loss}
                else:
                    # needed in the trainer otherwise (WtP-related only)
                    training_args.adapter_warmup_steps = args.adapter_warmup_steps
                    training_args.adapter_lr_multiplier = args.adapter_lr_multiplier
                    kwargs = {}

                with training_args.main_process_first():
                    logger.warning(model.backbone.adapter_summary())

                if args.freeze_classifier:
                    for n, p in model.backbone.named_parameters():
                        if "classifier" in n:
                            p.requires_grad = False
                # not done: keeping clf head is much better
                if args.clf_from_scratch:
                    model.backbone.classifier = torch.nn.Linear(model.backbone.config.hidden_size, num_labels)

                # eval only 5x during the entire training
                training_args.evaluation_strategy = "steps"
                training_args.eval_steps = max(
                    (
                        len(train_dataset)
                        // training_args.per_device_train_batch_size
                        // training_args.gradient_accumulation_steps
                        // args.eval_every
                        * training_args.num_train_epochs
                    ),
                    args.eval_every,
                )

                trainer_cls = AdapterTrainer if adapter_args.train_adapter else Trainer

                trainer = trainer_cls(
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
                        tokenizer=tokenizer,
                        add_lang_ids=False,
                    ),
                    **kwargs,
                )
                trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
                logger.warning(f"Finished training for {lang} {dataset_name}.")

                # only save trained module
                if training_args.local_rank == 0:
                    if not os.path.exists(os.path.join(training_args.output_dir, dataset_name, lang)):
                        os.makedirs(os.path.join(training_args.output_dir, dataset_name, lang))
                    save_model = copy.deepcopy(model.backbone)
                    save_model = save_model.to("cpu")
                    if adapter_args.train_adapter:
                        save_model.to("cpu").save_adapter(
                            adapter_name="text",
                            save_directory=os.path.join(training_args.output_dir, dataset_name, lang),
                            with_head=True,
                        )
                    else:
                        save_model.to("cpu").save_pretrained(os.path.join(training_args.output_dir, dataset_name, lang))


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
