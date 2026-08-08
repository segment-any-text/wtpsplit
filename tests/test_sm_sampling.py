from collections import Counter
from itertools import islice

from torch.utils.data import ConcatDataset, TensorDataset
import torch

from wtpsplit.train.sm_sampling import (
    DistributedWeightedGroupBatchSampler,
    WeightedGroupConcatDataset,
    smooth_weighted_cycle,
)


def test_smooth_weighted_cycle_is_exact_and_interleaved():
    schedule = list(islice(smooth_weighted_cycle([0.75, 0.25]), 40))
    assert Counter(schedule) == {0: 30, 1: 10}
    assert max(
        len(run)
        for run in "".join(map(str, schedule)).replace("1", " ").split()
    ) <= 3


def test_weighted_dataset_normalizes_group_weights():
    dataset = WeightedGroupConcatDataset(
        [TensorDataset(torch.arange(5)), TensorDataset(torch.arange(3))],
        [3, 1],
    )
    assert len(dataset) == 8
    assert dataset.group_weights == (0.75, 0.25)


def test_weighted_dataset_retains_language_subgroup_lengths():
    primary = ConcatDataset(
        [TensorDataset(torch.arange(4)), TensorDataset(torch.arange(12))]
    )
    replay = ConcatDataset(
        [TensorDataset(torch.arange(8)), TensorDataset(torch.arange(16))]
    )
    dataset = WeightedGroupConcatDataset([primary, replay], [0.5, 0.5])
    assert dataset.group_lengths == ((4, 12), (8, 16))


def test_weighted_batch_sampler_draws_whole_batches_at_fixed_fraction():
    sampler = DistributedWeightedGroupBatchSampler(
        lengths=[20, 20],
        weights=[0.5, 0.5],
        batch_size=2,
        rank=0,
        num_replicas=1,
        shuffle=False,
    )
    batches = list(islice(iter(sampler), 8))
    groups = [0 if max(batch) < 20 else 1 for batch in batches]
    assert groups == [0, 1, 0, 1, 0, 1, 0, 1]
    assert all(
        all(index < 20 for index in batch)
        or all(index >= 20 for index in batch)
        for batch in batches
    )


def test_weighted_batch_sampler_round_robins_languages_within_each_corpus():
    sampler = DistributedWeightedGroupBatchSampler(
        lengths=[16, 24],
        weights=[0.5, 0.5],
        batch_size=2,
        rank=0,
        num_replicas=1,
        shuffle=False,
        subgroup_lengths=((4, 12), (8, 16)),
    )
    batches = list(islice(iter(sampler), 8))
    primary_batches = batches[::2]
    replay_batches = batches[1::2]
    assert [batch[0] in range(0, 4) for batch in primary_batches] == [
        True,
        False,
        True,
        False,
    ]
    assert [batch[0] in range(16, 24) for batch in replay_batches] == [
        True,
        False,
        True,
        False,
    ]
