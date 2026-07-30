import gc
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import transformers
from datasets import Dataset
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

import wandb
from wtpsplit.char_head import CharacterDataCollator
from wtpsplit.evaluation.diagnostics.boundary_ceiling import dominant_script, separator_for
from wtpsplit.models import SubwordXLMForTokenClassification
from wtpsplit.train.backbones import resolve_backbone
from wtpsplit.train.sm_data import prepare_sentence_datasets
from wtpsplit.utils import Constants


@dataclass
class Args:
    block_size: int = 256
    num_layers: int = 12  # number of layers
    lim_lookahead: bool = False  # our "Lookahead" ablation
    without_pretraining: bool = False  # our "No pre-training" ablation
    no_sm_corruption: bool = False  # our "Only clean text" ablation
    use_character_head: bool = False
    balance_character_loss: bool = True
    # Identity preserves a trained token classifier; raw encoders should use random.
    character_head_init: str = "identity"
    run_final_evaluation: bool = False
    evaluation_only: bool = False
    # Comma-separated optimizer steps to retain as model-only learning-curve
    # checkpoints. This avoids saving every short interval during long pilots.
    checkpoint_milestones: str = None
    # Set these to train a backbone other than SaT/XLM-R (e.g. mmBERT for SaT 2). When
    # `model_name_or_path` is given it overrides the stage-1 checkpoint that would
    # otherwise be derived from `num_layers` and `lim_lookahead`.
    model_name_or_path: str = None
    tokenizer_name_or_path: str = None
    data_path: str = "data/all_data_11_05-all.pth"
    # Stage 2 configs should set this to "projected". The default preserves the
    # Stage 3 priority ud -> opus100 -> nllb, with projected as a final fallback.
    training_dataset: str = None
    # Optional bounded-data controls for smoke tests. Production defaults retain every
    # language and every training sentence.
    languages: str = None  # comma-separated language codes
    max_train_sentences_per_dataset: int = None
    max_eval_instances_per_dataset: int = 200
    # Total lookahead budget, divided across layers (see SubwordXLMConfig.lookahead).
    # Only applies on the `model_name_or_path` route; the SaT checkpoints below already
    # bake their lookahead setting into the pretrained weights.
    lookahead: int = None


# Parsing command line arguments or JSON config files as needed
parser = HfArgumentParser([Args, TrainingArguments])

if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
    args, training_args = parser.parse_json_file(sys.argv[1])
else:
    args, training_args = parser.parse_args_into_dataclasses()

# Model/head initialisation happens before Trainer is constructed, so Trainer's
# internal seed setup is too late for raw-backbone comparisons. Seed every RNG
# before loading the model and before stochastic sentence packing.
if training_args.full_determinism:
    transformers.enable_full_determinism(training_args.seed)
else:
    transformers.set_seed(training_args.seed)

data_path = args.data_path
all_data = torch.load(data_path, weights_only=True)

block_size = args.block_size
selected_languages = set(args.languages.split(",")) if args.languages else None

if args.max_train_sentences_per_dataset is not None and args.max_train_sentences_per_dataset < 1:
    raise ValueError("`max_train_sentences_per_dataset` must be at least 1.")
if args.max_eval_instances_per_dataset < 1:
    raise ValueError("`max_eval_instances_per_dataset` must be at least 1.")

checkpoint_milestones = (
    {int(step) for step in args.checkpoint_milestones.split(",")}
    if args.checkpoint_milestones
    else set()
)
if any(step < 1 for step in checkpoint_milestones):
    raise ValueError("`checkpoint_milestones` must contain positive optimizer steps.")
if checkpoint_milestones and max(checkpoint_milestones) > training_args.max_steps:
    raise ValueError("`checkpoint_milestones` cannot exceed `max_steps`.")


punct_chars = set(Constants.PUNCTUATION_CHARS)

train_sentences, test_sentences = prepare_sentence_datasets(
    all_data,
    selected_languages=selected_languages,
    requested_training_dataset=args.training_dataset,
    no_sm_corruption=args.no_sm_corruption,
    max_train_sentences_per_dataset=args.max_train_sentences_per_dataset,
    max_eval_instances_per_dataset=args.max_eval_instances_per_dataset,
)


def evaluation_script_sample(datasets, limit=2000):
    """Collect enough raw evaluation text to classify a language's script."""

    parts = []
    length = 0
    for instances in datasets.values():
        for instance in instances:
            sentences = [instance] if isinstance(instance, str) else instance
            for sentence in sentences:
                if not isinstance(sentence, str):
                    continue
                parts.append(sentence)
                length += len(sentence)
                if length >= limit:
                    return "".join(parts)[:limit]
    return "".join(parts)


# Token-head Arrow datasets omit raw text, while character-head datasets retain
# it for collation. Resolve scripts before packing so the final report has the
# same language/script grouping for every 2x2 arm.
evaluation_scripts = {
    lang_code: dominant_script(evaluation_script_sample(datasets))
    for lang_code, datasets in test_sentences.items()
}

# The serialized corpus is large and every selected sentence is now referenced by the
# bounded train/test dictionaries. Release the original nested dictionary before building
# token blocks so smoke runs do not retain an unnecessary 843 MB object graph.
del all_data
gc.collect()


tokenizer_checkpoint = args.tokenizer_name_or_path or "facebookAI/xlm-roberta-base"

if args.model_name_or_path:
    # Explicit checkpoint wins; this is the path used for non-XLM-R backbones, where the
    # SaT stage-1 checkpoint names below do not apply.
    model_checkpoint = args.model_name_or_path
elif args.without_pretraining:
    model_checkpoint = "facebookAI/xlm-roberta-base"
elif args.num_layers == 1:
    if not args.lim_lookahead:
        model_checkpoint = "segment-any-text/sat-1l-no-limited-lookahead"
    else:
        model_checkpoint = "segment-any-text/sat-1l"
elif args.num_layers == 3:
    if not args.lim_lookahead:
        model_checkpoint = "segment-any-text/sat-3l-no-limited-lookahead"
    else:
        model_checkpoint = "segment-any-text/sat-3"
elif args.num_layers == 6:
    if not args.lim_lookahead:
        model_checkpoint = "segment-any-text/sat-6l-no-limited-lookahead"
    else:
        model_checkpoint = "segment-any-text/sat-6l"
elif args.num_layers == 9:
    if not args.lim_lookahead:
        model_checkpoint = "segment-any-text/sat-9l-no-limited-lookahead"
    else:
        model_checkpoint = "segment-any-text/sat-9l"
elif args.num_layers == 12:
    if not args.lim_lookahead:
        model_checkpoint = "segment-any-text/sat-12l-no-limited-lookahead"
    else:
        model_checkpoint = "segment-any-text/sat-12l"
else:
    raise ValueError("Invalid number of layers. Valid values are 1, 3, 6, 9, 12.")

print(model_checkpoint)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

if args.model_name_or_path:
    # Non-SaT backbone (e.g. mmBERT). Pick the wrapper that understands lookahead rather
    # than letting Auto* hand back the stock transformers class, and trim the pretrained
    # stack to the requested depth the same way the SaT ladder is built from XLM-R.
    _, model_class, _ = resolve_backbone(model_checkpoint)
    model = model_class.from_pretrained(
        model_checkpoint,
        num_labels=1,
        ignore_mismatched_sizes=True,
        num_hidden_layers=args.num_layers,
        lookahead=args.lookahead,
        use_character_head=args.use_character_head,
        balance_character_loss=args.balance_character_loss,
        character_head_init=args.character_head_init,
    )
elif args.num_layers == 3 and args.without_pretraining:
    # special case for one of our ablations, where we trim XLM-R (without any of our newline pretraining) to 3 layers
    model = SubwordXLMForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=1,
        ignore_mismatched_sizes=True,
        num_hidden_layers=3,
        use_character_head=args.use_character_head,
        balance_character_loss=args.balance_character_loss,
        character_head_init=args.character_head_init,
    )
else:
    model = SubwordXLMForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=1,
        ignore_mismatched_sizes=True,
        use_character_head=args.use_character_head,
        balance_character_loss=args.balance_character_loss,
        character_head_init=args.character_head_init,
    )


def tokenize_and_get_labels(sentences, lang_code, dataset_name):
    # Stage 2 uses ISO language/script identifiers (for example `zho_Hans` and
    # `bod_Tibt`) that are absent from the legacy 89-language separator table.
    # Resolve their ISO-639-1/macrolanguage where possible and use the observed
    # script only as a fallback. This preserves Thai's sentence-boundary spaces
    # while avoiding artificial spaces for CJK, Khmer, Myanmar, and Tibetan.
    separator = separator_for(lang_code, dominant_script("".join(sentences)))

    joined_sentence = ""
    sentence_start_positions = []
    current_position = 0

    for sentence in sentences:
        if random.random() < 0.1 and sentence[-1] in punct_chars and dataset_name == "corrupted-social-media":
            if separator == " ":
                separator_used = ""
            else:
                separator_used = " "
        else:
            separator_used = separator

        if joined_sentence:
            joined_sentence += separator_used
            current_position += len(separator_used)
        start_position = current_position
        joined_sentence += sentence
        current_position += len(sentence)
        sentence_start_positions.append(start_position + len(sentence) - 1)

    tokenized_input = tokenizer(
        joined_sentence,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )

    offsets = tokenized_input["offset_mapping"]
    if args.use_character_head:
        # The final split is free because decoding always appends the document tail.
        labels = [0] * len(joined_sentence)
        for position in sentence_start_positions[:-1]:
            labels[position] = 1
    else:
        tokens = tokenized_input.tokens()
        labels = [0] * len(tokens)
        labels[-1] = 1
        sentence_index = 0

        for i in range(len(offsets)):
            if offsets[i][0] > sentence_start_positions[sentence_index]:
                labels[i - 1] = 1
                sentence_index += 1

    # Derived from the tokenizer, not hardcoded: XLM-R uses cls=0/sep=2 but mmBERT uses
    # cls=2/sep=1, so literals here would mislabel every chunk without raising.
    input_ids = [tokenizer.cls_token_id] + tokenized_input["input_ids"] + [tokenizer.sep_token_id]
    if not args.use_character_head:
        labels = [0] + labels + [0]

    return input_ids, labels, joined_sentence, offsets


def pack_sentences(input_data_dict, block_size, *, show_progress=True):
    def empty_columns():
        columns = {"input_ids": [], "attention_mask": [], "labels": []}
        if args.use_character_head:
            columns.update({"text": [], "offset_mapping": []})
        return columns

    packed_data = defaultdict(lambda: defaultdict(empty_columns))

    def append_block(lang_code, dataset_name, sentences):
        input_ids, labels, text, offsets = tokenize_and_get_labels(sentences, lang_code, dataset_name)
        num_to_pad = block_size - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * num_to_pad
        input_ids += [tokenizer.pad_token_id] * num_to_pad
        if not args.use_character_head:
            labels += [-100] * num_to_pad

        assert len(input_ids) == block_size, len(input_ids)
        if not args.use_character_head:
            assert len(input_ids) == len(labels), (len(input_ids), len(labels))
        else:
            assert len(labels) == len(text), (len(labels), len(text))

        columns = packed_data[lang_code][dataset_name]
        columns["input_ids"].append(input_ids)
        columns["attention_mask"].append(attention_mask)
        columns["labels"].append(labels)
        if args.use_character_head:
            columns["text"].append(text)
            columns["offset_mapping"].append(offsets)

    for lang_code in tqdm(input_data_dict, disable=not show_progress):
        for dataset_name, sentences in input_data_dict[lang_code].items():
            if dataset_name == "corrupted-social-media":
                p_add_to_block = 0.5
            else:
                p_add_to_block = 1.0

            token_count, one_block_sentences = 0, []

            for sentence in sentences:
                num_sentence_tokens = len(tokenizer(sentence, add_special_tokens=False)["input_ids"])

                if not sentence or sentence.isnumeric() or num_sentence_tokens == 0:
                    continue

                if token_count + num_sentence_tokens < block_size - 4 and (
                    random.random() <= p_add_to_block or len(one_block_sentences) == 0
                ):
                    one_block_sentences.append(sentence)
                    token_count += num_sentence_tokens
                else:
                    if one_block_sentences:
                        append_block(lang_code, dataset_name, one_block_sentences)

                    if num_sentence_tokens > block_size - 4:
                        one_block_sentences = []
                        token_count = 0
                    else:
                        one_block_sentences = [sentence]
                        token_count = num_sentence_tokens

            if one_block_sentences:
                append_block(lang_code, dataset_name, one_block_sentences)

            assert len(packed_data[lang_code][dataset_name]["input_ids"]) == len(
                packed_data[lang_code][dataset_name]["labels"]
            )

    return packed_data


def pack_language_datasets(input_data_dict, block_size, *, description):
    """Pack and Arrow-convert one language at a time to bound Python-object overlap."""
    datasets_by_language = {}
    for lang_code in tqdm(list(input_data_dict), desc=description):
        language_sentences = input_data_dict.pop(lang_code)
        packed_language = pack_sentences(
            {lang_code: language_sentences},
            block_size,
            show_progress=False,
        )[lang_code]
        datasets_by_language[lang_code] = {
            dataset_name: Dataset.from_dict(columns)
            for dataset_name, columns in packed_language.items()
        }
        del language_sentences, packed_language
        gc.collect()
    return datasets_by_language


if args.evaluation_only:
    train_dataset_by_language = {}
    del train_sentences
    gc.collect()
else:
    train_dataset_by_language = pack_language_datasets(
        train_sentences,
        block_size,
        description="Packing train languages",
    )
    del train_sentences
    gc.collect()

# Final evaluation is intentionally packed after training. Keeping the full packed
# train and evaluation corpora alive together pushed this workstation to roughly
# 7 GB RSS and left too little headroom while Transformers serialized a checkpoint.
# W&B's during-training callback still needs eager evaluation datasets.
use_wandb = "wandb" in training_args.report_to
needs_training_evaluation = use_wandb and not args.evaluation_only
if needs_training_evaluation:
    test_dataset = pack_language_datasets(
        test_sentences,
        block_size,
        description="Packing evaluation languages",
    )
    del test_sentences
    gc.collect()
    print("dataset_lifecycle evaluation_packing=eager")
else:
    test_dataset = None
    print("dataset_lifecycle evaluation_packing=deferred")
    if not args.run_final_evaluation:
        del test_sentences
        gc.collect()

experiment_name = model_checkpoint.split("/")[-1]

if args.no_sm_corruption:
    experiment_name += "-no-corruption"


def compute_prf(true_values, predicted_values):
    TP = np.sum((predicted_values == 1) & (true_values == 1))
    FP = np.sum((predicted_values == 1) & (true_values == 0))
    FN = np.sum((predicted_values == 0) & (true_values == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def sigmoid_array(x):
    # Extreme masked-BCE logits are expected after training. Clipping only
    # changes values that already round to 0/1 while avoiding exp overflow.
    return 1 / (1 + np.exp(-np.clip(x, -80, 80)))


def compute_metrics(p):
    predictions, labels = p

    predictions = np.reshape(predictions, (-1,))
    labels = np.reshape(labels, (-1,))

    predictions = sigmoid_array(predictions)

    predictions = predictions[labels != -100]
    labels = labels[labels != -100]

    threshold = 0.25

    preds = (predictions > threshold).astype(int)

    precision, recall, f1 = compute_prf(labels, preds)

    output_dict = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    return output_dict


class MultiDatasetEvalCallback(TrainerCallback):
    def __init__(self, eval_datasets):
        self.eval_datasets = eval_datasets

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        pass

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.eval_steps == 0:
            for lang_code in self.eval_datasets:
                for dataset_name, eval_dataset in self.eval_datasets[lang_code].items():
                    metrics = trainer.evaluate(eval_dataset)
                    for metric, result in metrics.items():
                        wandb.log(
                            {
                                f"eval/{dataset_name}/{lang_code}/{metric}": result,
                                "train/global_step": state.global_step,
                            }
                        )


class MilestoneSaveCallback(TrainerCallback):
    """Request checkpoints only at explicitly selected optimizer steps."""

    def __init__(self, milestones):
        self.milestones = frozenset(milestones)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.milestones:
            control.should_save = True
        return control


if args.evaluation_only:
    train_datasets = None
else:
    train_datasets = []
    for language_data in train_dataset_by_language.values():
        train_datasets.extend(language_data.values())
    random.shuffle(train_datasets)
    train_datasets = ConcatDataset(train_datasets)
del train_dataset_by_language
gc.collect()

run = wandb.init(project="sentence") if use_wandb else None
if run is not None:
    run.name = experiment_name
if args.use_character_head:
    # The collator consumes raw text and offset mappings before the batch reaches the
    # model; Trainer must not discard those non-forward columns first.
    training_args.remove_unused_columns = False

# args = TrainingArguments(
#     output_dir=experiment_name,
#     overwrite_output_dir=True,
#     evaluation_strategy="steps",
#     eval_steps=250,
#     report_to="wandb",
#     learning_rate=3e-5,
#     warmup_steps=500,
#     per_device_train_batch_size=128,
#     per_device_eval_batch_size=128,
#     weight_decay=0.01,
#     push_to_hub=False,
#     save_total_limit=1,
#     save_strategy="steps",
#     save_steps=1000,
#     load_best_model_at_end=False,
#     max_steps=20000,
# )


class RoundRobinSampler:
    def __init__(self, samplers: Sequence[Iterable], reinit: bool = False):
        self.samplers = samplers
        self.reinit = reinit

    def __iter__(self):
        iterators = [iter(sampler) for sampler in self.samplers]

        for i in cycle(range(len(iterators))):
            it = iterators[i]

            try:
                yield next(it)

            except StopIteration:
                if not self.reinit:
                    break

                it = iter(self.samplers[i])
                iterators[i] = it
                yield next(it)


def get_subset(length: int, i: int, k: int, offset: int = 0) -> Tuple[int, int]:
    assert i < k
    s = math.ceil(length / k)  # size of one split
    start = i * s
    end = min((i + 1) * s, length)
    return offset + start, offset + end


class DistributedRoundRobinBatchSampler:
    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        rank: int,
        num_replicas: int,
        drop_last: bool = False,
        seed: int = 0,
        shuffle: bool = True,
        reinit: bool = False,
    ):
        self.lengths = lengths
        offsets = [sum(lengths[:i]) for i in range(len(lengths))]
        self.ranges = [get_subset(length, rank, num_replicas, offset) for offset, length in zip(offsets, lengths)]
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.reinit = reinit
        self.batch_size = batch_size
        self.batch_start = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        batch_samplers = [
            BatchSampler(
                (SubsetRandomSampler(range(start, end), generator=g) if self.shuffle else range(start, end)),
                self.batch_size,
                self.drop_last,
            )
            for (start, end) in self.ranges
        ]

        sampler = RoundRobinSampler(batch_samplers, reinit=self.reinit)
        return iter(sampler)

    def __len__(self):
        return min(length for length in self.lengths) // self.batch_size


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The SaT wrappers accept ``**kwargs`` to support backbone-specific inputs, but
        # they do not consume Transformers 5's loss-normalisation kwarg. If left at the
        # inferred default, Trainer forwards it through the wrapper into the backbone.
        self.model_accepts_loss_kwargs = False

    def get_train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset

        if isinstance(dataset, ConcatDataset):
            sizes = [len(ds) for ds in dataset.datasets]
        else:
            sizes = [len(dataset)]

        loader = DataLoader(
            dataset,
            batch_sampler=DistributedRoundRobinBatchSampler(
                lengths=sizes,
                batch_size=self.args.train_batch_size,
                drop_last=False,
                rank=self.args.process_index,
                num_replicas=self.args.world_size,
                seed=self.args.seed,
                reinit=True,
            ),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            collate_fn=self.data_collator,
        )
        return loader


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_datasets,
    eval_dataset=None,
    compute_metrics=compute_metrics,
    data_collator=CharacterDataCollator() if args.use_character_head else None,
    processing_class=tokenizer,
    callbacks=[
        *([MultiDatasetEvalCallback(test_dataset)] if needs_training_evaluation else []),
        *([MilestoneSaveCallback(checkpoint_milestones)] if checkpoint_milestones else []),
    ],
)

if not args.evaluation_only:
    trainer.train()
    # Checkpointing is complete. Final prediction needs only the model and collator,
    # so release the full training corpus and optimizer before packing held-out data.
    trainer.train_dataset = None
    trainer.optimizer = None
    trainer.lr_scheduler = None
    train_datasets = None
    model.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("dataset_lifecycle training_state=released")

if torch.cuda.is_available():
    gib = 1024**3
    print(
        "cuda_peak_memory "
        f"allocated={torch.cuda.max_memory_allocated() / gib:.3f}GiB "
        f"reserved={torch.cuda.max_memory_reserved() / gib:.3f}GiB"
    )

if args.run_final_evaluation:
    if test_dataset is None:
        test_dataset = pack_language_datasets(
            test_sentences,
            block_size,
            description="Packing evaluation languages",
        )
        del test_sentences
        gc.collect()
        print("dataset_lifecycle evaluation_packing=materialized")

    all_probabilities = []
    all_labels = []
    language_probabilities = defaultdict(list)
    language_labels = defaultdict(list)
    language_scripts = defaultdict(Counter)
    dataset_count = 0
    for lang_code in test_dataset:
        for eval_dataset in test_dataset[lang_code].values():
            prediction = trainer.predict(eval_dataset)
            logits = np.asarray(prediction.predictions).reshape(-1)
            labels = np.asarray(prediction.label_ids).reshape(-1)
            keep = labels != -100
            kept_probabilities = sigmoid_array(logits[keep])
            kept_labels = labels[keep]
            all_probabilities.append(kept_probabilities)
            all_labels.append(kept_labels)
            language_probabilities[lang_code].append(kept_probabilities)
            language_labels[lang_code].append(kept_labels)
            language_scripts[lang_code][evaluation_scripts[lang_code]] += len(kept_labels)
            dataset_count += 1

    probabilities = np.concatenate(all_probabilities)
    labels = np.concatenate(all_labels)

    def metrics_at(group_probabilities, group_labels, threshold):
        precision, recall, f1 = compute_prf(
            group_labels,
            (group_probabilities > threshold).astype(int),
        )
        return {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    thresholds = np.unique(
        np.concatenate(
            [
                np.geomspace(1e-4, 1e-2, 20),
                np.linspace(0.01, 0.99, 99),
            ]
        )
    )

    def summarize_group(group_probabilities, group_labels):
        swept = [
            metrics_at(group_probabilities, group_labels, threshold)
            for threshold in thresholds
        ]
        return {
            "positions": int(group_labels.size),
            "positives": int(group_labels.sum()),
            "fixed_0.25": metrics_at(group_probabilities, group_labels, 0.25),
            "best": max(swept, key=lambda result: result["f1"]),
        }

    global_metrics = summarize_group(probabilities, labels)
    global_threshold = global_metrics["best"]["threshold"]
    languages = {}
    script_probabilities = defaultdict(list)
    script_labels = defaultdict(list)
    for lang_code in sorted(language_probabilities):
        lang_probabilities = np.concatenate(language_probabilities[lang_code])
        lang_labels = np.concatenate(language_labels[lang_code])
        script = (
            language_scripts[lang_code].most_common(1)[0][0]
            if language_scripts[lang_code]
            else "UNKNOWN"
        )
        languages[lang_code] = {
            "script": script,
            **summarize_group(lang_probabilities, lang_labels),
            "at_global_best": metrics_at(
                lang_probabilities,
                lang_labels,
                global_threshold,
            ),
        }
        script_probabilities[script].append(lang_probabilities)
        script_labels[script].append(lang_labels)

    scripts = {}
    for script in sorted(script_probabilities):
        group_probabilities = np.concatenate(script_probabilities[script])
        group_labels = np.concatenate(script_labels[script])
        scripts[script] = {
            **summarize_group(group_probabilities, group_labels),
            "at_global_best": metrics_at(
                group_probabilities,
                group_labels,
                global_threshold,
            ),
        }

    language_scores = [
        result["at_global_best"]
        for result in languages.values()
        if result["positives"] > 0
    ]
    macro_language_at_global_best = {
        metric: float(np.mean([score[metric] for score in language_scores]))
        for metric in ("precision", "recall", "f1")
    }
    macro_language_at_global_best["threshold"] = global_threshold

    detailed_summary = {
        "datasets": dataset_count,
        **global_metrics,
        "macro_language_at_global_best": macro_language_at_global_best,
        "scripts": scripts,
        "languages": languages,
    }
    output_path = Path(training_args.output_dir) / "final_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(detailed_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        key: value
        for key, value in detailed_summary.items()
        if key != "languages"
    }
    summary["languages"] = len(languages)
    summary["details_path"] = str(output_path)
    print(f"final_eval {json.dumps(summary, sort_keys=True)}")
