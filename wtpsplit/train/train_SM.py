import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import cycle
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import transformers
from datasets import Dataset
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, Trainer, TrainerCallback, TrainingArguments

import wandb
from wtpsplit.models import SubwordXLMForTokenClassification
from wtpsplit.utils import Constants


@dataclass
class Args:
    block_size: int = 256
    num_layers: int = 12  # number of layers
    lim_lookahead: bool = False  # our "Lookahead" ablation
    without_pretraining: bool = False  # our "No pre-training" ablation
    no_sm_corruption: bool = False  # our "Only clean text" ablation


# Parsing command line arguments or JSON config files as needed
parser = HfArgumentParser([Args, TrainingArguments])

if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
    args, training_args = parser.parse_json_file(sys.argv[1])
else:
    args, training_args = parser.parse_args_into_dataclasses()

data_path = "data/all_data_11_05-all.pth"
all_data = torch.load(data_path)

block_size = args.block_size

train_sentences = defaultdict(lambda: defaultdict(list))
test_sentences = defaultdict(lambda: defaultdict(list))

punct_chars = set(Constants.PUNCTUATION_CHARS)


for lang_code in tqdm(all_data, desc="Loading data"):
    if "-" in lang_code or "_" in lang_code:
        # we only train on monolingual data in SM, so no "en-de" code-switching for example!
        pass
    elif (
        "ud" in all_data[lang_code]["sentence"]
        and all_data[lang_code]["sentence"]["ud"]["meta"]["train_data"] is not None
    ):
        train_data = all_data[lang_code]["sentence"]["ud"]["meta"]["train_data"]

        if len(train_data) < 10000:
            # some languages have an insufficient number of sentences to fill a single batch
            # this is just a quick way to upsample these so we don't run into problems later
            # later we will use a uniform round-robin sampler for all languages
            train_data = train_data * (10000 // len(train_data) + 1)

        train_sentences[lang_code]["uncorrupted"].extend(train_data)

        if not args.no_sm_corruption:
            train_data = all_data[lang_code]["sentence"]["ud-corrupted-asr"]["meta"]["train_data"]

            if len(train_data) < 5000:
                # some languages have an insufficient number of sentences to fill a single batch
                # this is just a quick way to upsample these so we don't run into problems later
                # later we will use a uniform round-robin sampler for all languages
                train_data = train_data * (10000 // len(train_data) + 1)

            train_sentences[lang_code]["corrupted-asr"].extend(train_data)

            train_data = all_data[lang_code]["sentence"]["ud-corrupted-social-media"]["meta"]["train_data"]

            if len(train_data) < 5000:
                # some languages have an insufficient number of sentences to fill a single batch
                # this is just a quick way to upsample these so we don't run into problems later
                # later we will use a uniform round-robin sampler for all languages
                train_data = train_data * (10000 // len(train_data) + 1)

            train_sentences[lang_code]["corrupted-social-media"].extend(train_data)

    elif (
        "opus100" in all_data[lang_code]["sentence"]
        and all_data[lang_code]["sentence"]["opus100"]["meta"]["train_data"] is not None
    ):
        train_data = all_data[lang_code]["sentence"]["opus100"]["meta"]["train_data"]
        train_sentences[lang_code]["uncorrupted"].extend(train_data)

        if not args.no_sm_corruption:
            train_data = all_data[lang_code]["sentence"]["opus100-corrupted-asr"]["meta"]["train_data"]
            train_sentences[lang_code]["corrupted-asr"].extend(train_data)

            train_data = all_data[lang_code]["sentence"]["opus100-corrupted-social-media"]["meta"]["train_data"]
            train_sentences[lang_code]["corrupted-social-media"].extend(train_data)
    else:
        train_data = all_data[lang_code]["sentence"]["nllb"]["meta"]["train_data"]
        train_sentences[lang_code]["uncorrupted"].extend(train_data)

        if not args.no_sm_corruption:
            train_data = all_data[lang_code]["sentence"]["nllb-corrupted-asr"]["meta"]["train_data"]
            train_sentences[lang_code]["corrupted-asr"].extend(train_data)

            train_data = all_data[lang_code]["sentence"]["nllb-corrupted-social-media"]["meta"]["train_data"]
            train_sentences[lang_code]["corrupted-social-media"].extend(train_data)

    for dataset in all_data[lang_code]["sentence"]:
        if any(dataset.startswith(x) for x in ["short-sequences", "legal"]):
            continue

        test_data = all_data[lang_code]["sentence"][dataset]["data"]
        test_sentences[lang_code][dataset].extend(test_data[:200])


tokenizer_checkpoint = "xlm-roberta-base"

if args.without_pretraining:
    model_checkpoint = "xlm-roberta-base"
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

if args.num_layers == 3 and args.without_pretraining:
    # special case for one of our ablations, where we trim XLM-R (without any of our newline pretraining) to 3 layers
    model = SubwordXLMForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=1,
        ignore_mismatched_sizes=True,
        num_hidden_layers=3,
    )
else:
    model = SubwordXLMForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )


def tokenize_and_get_labels(sentences, lang_code, dataset_name):
    separator = Constants.SEPARATORS.get(lang_code, " ")

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

    tokens = tokenized_input.tokens()
    offsets = tokenized_input["offset_mapping"]
    sentence_ending_labels = [0] * len(tokens)

    sentence_ending_labels[-1] = 1
    sentence_index = 0

    for i in range(len(offsets)):
        if offsets[i][0] > sentence_start_positions[sentence_index]:
            sentence_ending_labels[i - 1] = 1
            sentence_index += 1

    input_ids = [0] + tokenized_input["input_ids"] + [2]
    labels = [0] + sentence_ending_labels + [0]

    return input_ids, labels


def pack_sentences(input_data_dict, block_size):
    packed_data = defaultdict(lambda: defaultdict(lambda: {"input_ids": [], "attention_mask": [], "labels": []}))

    for lang_code in tqdm(input_data_dict):
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
                        input_ids, labels = tokenize_and_get_labels(one_block_sentences, lang_code, dataset_name)

                        num_to_pad = block_size - len(input_ids)
                        attention_mask = [1] * len(input_ids) + [0] * num_to_pad
                        input_ids += [tokenizer.pad_token_id] * num_to_pad
                        labels += [-100] * num_to_pad

                        assert len(input_ids) == block_size, len(input_ids)
                        assert len(input_ids) == len(labels), (
                            len(input_ids),
                            len(labels),
                        )

                        packed_data[lang_code][dataset_name]["input_ids"].append(input_ids)
                        packed_data[lang_code][dataset_name]["attention_mask"].append(attention_mask)
                        packed_data[lang_code][dataset_name]["labels"].append(labels)

                    if num_sentence_tokens > block_size - 4:
                        one_block_sentences = []
                        token_count = 0
                    else:
                        one_block_sentences = [sentence]
                        token_count = num_sentence_tokens

            if one_block_sentences:
                input_ids, labels = tokenize_and_get_labels(one_block_sentences, lang_code, dataset_name)

                num_to_pad = block_size - len(input_ids)
                attention_mask = [1] * len(input_ids) + [0] * num_to_pad
                input_ids += [tokenizer.pad_token_id] * num_to_pad
                labels += [-100] * num_to_pad

                assert len(input_ids) == block_size, len(input_ids)
                assert len(input_ids) == len(labels), (len(input_ids), len(labels))

                packed_data[lang_code][dataset_name]["input_ids"].append(input_ids)
                packed_data[lang_code][dataset_name]["attention_mask"].append(attention_mask)
                packed_data[lang_code][dataset_name]["labels"].append(labels)

            assert len(packed_data[lang_code][dataset_name]["input_ids"]) == len(
                packed_data[lang_code][dataset_name]["labels"]
            )

    return packed_data


packed_train_data = pack_sentences(train_sentences, block_size)
packed_test_data = pack_sentences(test_sentences, block_size)
test_dataset = {lang_code: defaultdict(dict) for lang_code in packed_test_data}

for lang_code in packed_test_data:
    for dataset_name in packed_test_data[lang_code]:
        test_dataset[lang_code][dataset_name] = Dataset.from_dict(packed_test_data[lang_code][dataset_name])

experiment_name = model_checkpoint.split("/")[-1]

if args.no_sm_corruption:
    experiment_name += "-no-corruption"

training_args.output_dir = experiment_name


def compute_prf(true_values, predicted_values):
    TP = np.sum((predicted_values == 1) & (true_values == 1))
    FP = np.sum((predicted_values == 1) & (true_values == 0))
    FN = np.sum((predicted_values == 0) & (true_values == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))


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


multi_dataset_eval_callback = MultiDatasetEvalCallback(test_dataset)

train_datasets = []

for lang_code in packed_train_data:
    for dataset_name in packed_train_data[lang_code]:
        train_datasets.append(Dataset.from_dict(packed_train_data[lang_code][dataset_name]))

random.shuffle(train_datasets)

train_datasets = ConcatDataset(train_datasets)

run = wandb.init(project="sentence")
wandb.run.name = experiment_name

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
    tokenizer=tokenizer,
    callbacks=[multi_dataset_eval_callback],
)

trainer.train()
