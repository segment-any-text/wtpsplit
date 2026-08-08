import gc
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import Dataset
from torch.utils.data import ConcatDataset, DataLoader
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
from wtpsplit.train.sm_arguments import SentenceTrainingArguments
from wtpsplit.train.sm_data import prepare_sentence_datasets
from wtpsplit.train.sm_sampling import (
    DistributedRoundRobinBatchSampler,
    DistributedWeightedGroupBatchSampler,
    WeightedGroupConcatDataset,
)
from wtpsplit.utils import Constants

# Parsing command line arguments or JSON config files as needed
parser = HfArgumentParser([SentenceTrainingArguments, TrainingArguments])

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
if not 0 <= args.replay_fraction < 1:
    raise ValueError("`replay_fraction` must be in the half-open interval [0, 1).")
if bool(args.replay_data_path) != (args.replay_fraction > 0):
    raise ValueError(
        "Set both `replay_data_path` and a positive `replay_fraction`, or neither."
    )
if (
    args.max_replay_train_sentences_per_dataset is not None
    and args.max_replay_train_sentences_per_dataset < 1
):
    raise ValueError("`max_replay_train_sentences_per_dataset` must be at least 1.")

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

replay_train_sentences = None
if args.replay_data_path:
    replay_data = torch.load(args.replay_data_path, weights_only=True)
    replay_languages = (
        set(args.replay_languages.split(","))
        if args.replay_languages
        else None
    )
    replay_train_sentences, _ = prepare_sentence_datasets(
        replay_data,
        selected_languages=replay_languages,
        requested_training_dataset=args.replay_training_dataset,
        no_sm_corruption=args.no_sm_corruption,
        max_train_sentences_per_dataset=(
            args.max_replay_train_sentences_per_dataset
        ),
        max_eval_instances_per_dataset=1,
    )
    del replay_data
    gc.collect()
    print(
        "replay "
        f"path={args.replay_data_path} fraction={args.replay_fraction:.4f} "
        f"languages={len(replay_train_sentences)}"
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
    replay_dataset_by_language = {}
    replay_train_sentences = None
    gc.collect()
else:
    train_dataset_by_language = pack_language_datasets(
        train_sentences,
        block_size,
        description="Packing train languages",
    )
    del train_sentences
    gc.collect()
    if replay_train_sentences is not None:
        replay_dataset_by_language = pack_language_datasets(
            replay_train_sentences,
            block_size,
            description="Packing replay languages",
        )
        del replay_train_sentences
        gc.collect()
    else:
        replay_dataset_by_language = {}

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
    # Transformers 5 requires eval_dataset whenever eval_strategy != "no".
    # During-training eval uses MultiDatasetEvalCallback only when W&B is on;
    # otherwise keep packing deferred and disable step/epoch eval.
    if training_args.eval_strategy != "no":
        training_args.eval_strategy = "no"
        print("dataset_lifecycle eval_strategy=forced_no (deferred packing)")

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
    # Cast explicitly: object/None arrays from Trainer predict break np.clip.
    x = np.asarray(x, dtype=np.float64)
    return 1 / (1 + np.exp(-np.clip(x, -80.0, 80.0)))


def _flatten_numeric(values, dtype):
    """Flatten Trainer predict outputs that may be ragged object arrays."""
    if values is None:
        return np.zeros(0, dtype=dtype)
    if isinstance(values, (tuple, list)):
        values = values[0]
    array = np.asarray(values)
    if array.dtype != object:
        return array.astype(dtype, copy=False).reshape(-1)
    pieces = []
    for item in array.flat:
        if item is None:
            continue
        pieces.append(np.asarray(item, dtype=dtype).reshape(-1))
    if not pieces:
        return np.zeros(0, dtype=dtype)
    return np.concatenate(pieces)


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
    primary_datasets = []
    for language_data in train_dataset_by_language.values():
        primary_datasets.extend(language_data.values())
    random.shuffle(primary_datasets)
    primary_dataset = ConcatDataset(primary_datasets)
    if replay_dataset_by_language:
        replay_datasets = []
        for language_data in replay_dataset_by_language.values():
            replay_datasets.extend(language_data.values())
        random.shuffle(replay_datasets)
        replay_dataset = ConcatDataset(replay_datasets)
        train_datasets = WeightedGroupConcatDataset(
            [primary_dataset, replay_dataset],
            [1 - args.replay_fraction, args.replay_fraction],
        )
        print(
            "training_mixture "
            f"primary_examples={len(primary_dataset)} "
            f"replay_examples={len(replay_dataset)} "
            f"replay_fraction={args.replay_fraction:.4f}"
        )
    else:
        train_datasets = primary_dataset
del train_dataset_by_language
del replay_dataset_by_language
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


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The SaT wrappers accept ``**kwargs`` to support backbone-specific inputs, but
        # they do not consume Transformers 5's loss-normalisation kwarg. If left at the
        # inferred default, Trainer forwards it through the wrapper into the backbone.
        self.model_accepts_loss_kwargs = False

    def get_train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset

        if isinstance(dataset, WeightedGroupConcatDataset):
            sizes = [len(group) for group in dataset.datasets]
            batch_sampler = DistributedWeightedGroupBatchSampler(
                lengths=sizes,
                weights=dataset.group_weights,
                batch_size=self.args.train_batch_size,
                drop_last=False,
                rank=self.args.process_index,
                num_replicas=self.args.world_size,
                seed=self.args.seed,
                subgroup_lengths=dataset.group_lengths,
            )
        elif isinstance(dataset, ConcatDataset):
            sizes = [len(ds) for ds in dataset.datasets]
            batch_sampler = DistributedRoundRobinBatchSampler(
                lengths=sizes,
                batch_size=self.args.train_batch_size,
                drop_last=False,
                rank=self.args.process_index,
                num_replicas=self.args.world_size,
                seed=self.args.seed,
                reinit=True,
            )
        else:
            sizes = [len(dataset)]
            batch_sampler = DistributedRoundRobinBatchSampler(
                lengths=sizes,
                batch_size=self.args.train_batch_size,
                drop_last=False,
                rank=self.args.process_index,
                num_replicas=self.args.world_size,
                seed=self.args.seed,
                reinit=True,
            )

        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
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
    # Downstream evaluation and the optional Stage 3 run consume the run's
    # output directory, not an internal checkpoint-* directory. Keep a final
    # loadable model and tokenizer at that stable path.
    trainer.save_model()
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
            if len(eval_dataset) == 0:
                continue
            prediction = trainer.predict(eval_dataset)
            logits = _flatten_numeric(prediction.predictions, np.float64)
            labels = _flatten_numeric(prediction.label_ids, np.int64)
            if logits.size != labels.size:
                raise ValueError(
                    f"Prediction/label length mismatch for {lang_code}: "
                    f"{logits.size} vs {labels.size}"
                )
            keep = labels != -100
            kept_probabilities = sigmoid_array(logits[keep])
            kept_labels = labels[keep]
            if kept_labels.size == 0:
                continue
            all_probabilities.append(kept_probabilities)
            all_labels.append(kept_labels)
            language_probabilities[lang_code].append(kept_probabilities)
            language_labels[lang_code].append(kept_labels)
            language_scripts[lang_code][evaluation_scripts[lang_code]] += len(kept_labels)
            dataset_count += 1

    if not all_probabilities:
        probabilities = np.zeros(0, dtype=np.float64)
        labels = np.zeros(0, dtype=np.int64)
    else:
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
