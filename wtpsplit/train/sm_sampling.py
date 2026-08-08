"""Batch samplers for language balancing and Stage 2 replay."""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import cycle
from typing import Iterable, Iterator, Sequence

import torch
from torch.utils.data import BatchSampler, ConcatDataset, SubsetRandomSampler


class WeightedGroupConcatDataset(ConcatDataset):
    """A concatenation whose children are sampled at declared batch weights."""

    def __init__(self, datasets: Sequence, weights: Sequence[float]):
        super().__init__(datasets)
        if len(datasets) != len(weights) or not datasets:
            raise ValueError("datasets and weights must be non-empty and have equal length")
        if any(weight <= 0 for weight in weights):
            raise ValueError("all group weights must be positive")
        total = float(sum(weights))
        self.group_weights = tuple(float(weight) / total for weight in weights)
        # Corpus groups normally contain one child dataset per language. Keep
        # those boundaries so replay does not change language balancing.
        self.group_lengths = tuple(
            tuple(len(child) for child in dataset.datasets)
            if isinstance(dataset, ConcatDataset)
            else (len(dataset),)
            for dataset in datasets
        )


class RoundRobinSampler:
    def __init__(self, samplers: Sequence[Iterable], reinit: bool = False):
        self.samplers = samplers
        self.reinit = reinit

    def __iter__(self):
        iterators = [iter(sampler) for sampler in self.samplers]
        for index in cycle(range(len(iterators))):
            iterator = iterators[index]
            try:
                yield next(iterator)
            except StopIteration:
                if not self.reinit:
                    break
                iterator = iter(self.samplers[index])
                iterators[index] = iterator
                yield next(iterator)


def get_subset(length: int, index: int, parts: int, offset: int = 0) -> tuple[int, int]:
    if not 0 <= index < parts:
        raise ValueError("distributed rank must be smaller than world size")
    size = math.ceil(length / parts)
    start = index * size
    end = min((index + 1) * size, length)
    return offset + start, offset + end


def smooth_weighted_cycle(weights: Sequence[float]) -> Iterator[int]:
    """Yield a deterministic, evenly interleaved weighted schedule forever."""

    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError("weights must be a non-empty sequence of positive values")
    fractions = [Fraction(str(weight)).limit_denominator(1000) for weight in weights]
    denominator = math.lcm(*(value.denominator for value in fractions))
    integers = [value.numerator * (denominator // value.denominator) for value in fractions]
    common = math.gcd(*integers)
    integers = [value // common for value in integers]
    total = sum(integers)
    current = [0] * len(integers)
    while True:
        for index, weight in enumerate(integers):
            current[index] += weight
        selected = max(range(len(current)), key=lambda index: (current[index], -index))
        current[selected] -= total
        yield selected


class DistributedRoundRobinBatchSampler:
    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        rank: int,
        num_replicas: int,
        drop_last: bool = False,
        seed: int = 0,
        shuffle: bool = True,
        reinit: bool = False,
    ):
        self.lengths = lengths
        offsets = [sum(lengths[:index]) for index in range(len(lengths))]
        self.ranges = [
            get_subset(length, rank, num_replicas, offset)
            for offset, length in zip(offsets, lengths)
        ]
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.reinit = reinit
        self.batch_size = batch_size

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        batch_samplers = [
            BatchSampler(
                SubsetRandomSampler(range(start, end), generator=generator)
                if self.shuffle
                else range(start, end),
                self.batch_size,
                self.drop_last,
            )
            for start, end in self.ranges
        ]
        return iter(RoundRobinSampler(batch_samplers, reinit=self.reinit))

    def __len__(self):
        return min(self.lengths) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class DistributedWeightedGroupBatchSampler:
    """Sample whole batches from corpus groups at fixed, deterministic weights."""

    def __init__(
        self,
        lengths: list[int],
        weights: Sequence[float],
        batch_size: int,
        rank: int,
        num_replicas: int,
        drop_last: bool = False,
        seed: int = 0,
        shuffle: bool = True,
        subgroup_lengths: Sequence[Sequence[int]] | None = None,
    ):
        if len(lengths) != len(weights):
            raise ValueError("one sampling weight is required for each corpus group")
        self.lengths = lengths
        self.weights = tuple(weights)
        if subgroup_lengths is None:
            subgroup_lengths = [(length,) for length in lengths]
        if len(subgroup_lengths) != len(lengths):
            raise ValueError("one subgroup-length sequence is required per corpus group")
        if any(sum(group) != length for group, length in zip(subgroup_lengths, lengths)):
            raise ValueError("subgroup lengths must sum to their corpus-group length")
        if any(not group for group in subgroup_lengths):
            raise ValueError("corpus groups must contain at least one subgroup")
        if any(length < num_replicas for group in subgroup_lengths for length in group):
            raise ValueError("each subgroup needs at least one example per distributed rank")

        group_offsets = [sum(lengths[:index]) for index in range(len(lengths))]
        self.group_ranges = []
        for group_offset, group in zip(group_offsets, subgroup_lengths):
            subgroup_offsets = [sum(group[:index]) for index in range(len(group))]
            self.group_ranges.append(
                [
                    get_subset(
                        length,
                        rank,
                        num_replicas,
                        group_offset + subgroup_offset,
                    )
                    for subgroup_offset, length in zip(subgroup_offsets, group)
                ]
            )
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        def new_iterator(group: int):
            batch_samplers = []
            for start, end in self.group_ranges[group]:
                sampler = (
                    SubsetRandomSampler(range(start, end), generator=generator)
                    if self.shuffle
                    else range(start, end)
                )
                batch_samplers.append(
                    BatchSampler(sampler, self.batch_size, self.drop_last)
                )
            return iter(RoundRobinSampler(batch_samplers, reinit=True))

        iterators = [new_iterator(group) for group in range(len(self.group_ranges))]
        schedule = smooth_weighted_cycle(self.weights)
        while True:
            group = next(schedule)
            try:
                yield next(iterators[group])
            except StopIteration:
                iterators[group] = new_iterator(group)
                yield next(iterators[group])

    def __len__(self):
        return sum(math.ceil(length / self.batch_size) for length in self.lengths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
