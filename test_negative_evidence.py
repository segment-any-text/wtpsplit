"""Tests for the negative-evidence term in constrained segmentation.

The shipped objective scores boundaries but never scores the decision not to place one,
so the length prior ends up as the only force creating splits. `use_negative_evidence`
adds the complement term. See V2_ROADMAP B2b.
"""

import numpy as np
import pytest

from wtpsplit.utils.constraints import constrained_segmentation
from wtpsplit.utils.priors import create_prior_function


def gaussian(target=20, spread=8, **kwargs):
    return create_prior_function("gaussian", {"target_length": target, "spread": spread, **kwargs})


def reference_score(probs, boundaries, prior_fn, use_negative_evidence):
    """Objective evaluated directly from its definition, for cross-checking the DP."""
    n = len(probs)
    cuts = [0, *boundaries, n]
    total = 0.0
    for start, end in zip(cuts, cuts[1:]):
        prior = prior_fn(end - start)
        if prior <= 0:
            return -np.inf
        total += np.log(prior)
    for b in boundaries:
        total += np.log(probs[b - 1])
    if use_negative_evidence:
        # candidate boundary positions are 0..n-2; the terminal boundary has no probability
        non_boundary = set(range(n - 1)) - {b - 1 for b in boundaries}
        total += sum(np.log1p(-probs[j]) for j in non_boundary)
    return total


def brute_force_best(probs, prior_fn, use_negative_evidence, min_length=1):
    """Exhaustive search over all segmentations, for small n."""
    from itertools import combinations

    n = len(probs)
    positions = range(1, n)
    best, best_score = None, -np.inf
    for r in range(len(positions) + 1):
        for combo in combinations(positions, r):
            cuts = [0, *combo, n]
            if any(b - a < min_length for a, b in zip(cuts, cuts[1:])):
                continue
            score = reference_score(probs, list(combo), prior_fn, use_negative_evidence)
            if score > best_score:
                best, best_score = list(combo), score
    return best, best_score


@pytest.mark.parametrize("seed", range(6))
def test_default_is_negative_evidence(seed):
    """The default objective includes the complement term as of 3.0.

    `target=10, spread=5` is chosen deliberately: with the looser prior the two objectives
    disagree on every seed, so this cannot pass vacuously. A tighter prior lets the length
    term dominate and both agree, which is how the previous version of this test kept
    passing after the default had already been flipped.
    """
    rng = np.random.default_rng(seed)
    probs = rng.random(40) * 0.5
    prior_fn = gaussian(target=10, spread=5)

    default = constrained_segmentation(probs, prior_fn)
    explicit_on = constrained_segmentation(probs, prior_fn, use_negative_evidence=True)
    explicit_off = constrained_segmentation(probs, prior_fn, use_negative_evidence=False)

    assert list(default) == list(explicit_on)
    assert list(default) != list(explicit_off), "prior too tight to distinguish the objectives"


@pytest.mark.parametrize("seed", range(6))
def test_legacy_objective_remains_reachable(seed):
    """Passing False must still reproduce the pre-3.0 boundary-only objective exactly.

    Length 11 because `brute_force_best` enumerates every boundary subset; the existing
    negative-evidence brute-force test uses the same bound for the same reason.
    """
    rng = np.random.default_rng(seed)
    probs = rng.random(11) * 0.6
    prior_fn = gaussian(target=4, spread=3)

    found = constrained_segmentation(probs, prior_fn, use_negative_evidence=False)
    expected, expected_score = brute_force_best(probs, prior_fn, use_negative_evidence=False, min_length=1)
    found_score = reference_score(probs, list(found), prior_fn, use_negative_evidence=False)

    assert found_score == pytest.approx(expected_score, abs=1e-9), f"{found} vs {expected}"


@pytest.mark.parametrize("seed", range(6))
def test_viterbi_matches_brute_force_with_negative_evidence(seed):
    """The DP must actually maximise the stated objective, not merely differ from before."""
    rng = np.random.default_rng(100 + seed)
    probs = rng.random(11) * 0.9 + 0.02
    prior_fn = gaussian(target=4, spread=3)

    found = constrained_segmentation(probs, prior_fn, use_negative_evidence=True)
    expected, expected_score = brute_force_best(probs, prior_fn, use_negative_evidence=True)

    found_score = reference_score(probs, list(found), prior_fn, use_negative_evidence=True)
    assert found_score == pytest.approx(expected_score, abs=1e-9), f"{found} vs {expected}"


@pytest.mark.parametrize("seed", range(4))
def test_logit_form_equals_explicit_complement_sum(seed):
    """log-odds per boundary must equal summing log(1-p) over every non-boundary.

    The two differ only by a segmentation-independent constant, so the argmax matches.
    """
    rng = np.random.default_rng(200 + seed)
    probs = rng.random(9) * 0.8 + 0.05
    prior_fn = gaussian(target=3, spread=2)

    boundaries = constrained_segmentation(probs, prior_fn, use_negative_evidence=True)

    # score every candidate segmentation under both formulations and compare rankings
    from itertools import combinations

    proper, logit_form = [], []
    for r in range(len(range(1, len(probs))) + 1):
        for combo in combinations(range(1, len(probs)), r):
            proper.append(reference_score(probs, list(combo), prior_fn, True))
            base = sum(np.log(prior_fn(b - a)) for a, b in zip([0, *combo], [*combo, len(probs)]))
            logit_form.append(base + sum(np.log(probs[b - 1]) - np.log1p(-probs[b - 1]) for b in combo))

    offset = np.array(proper) - np.array(logit_form)
    assert np.allclose(offset, offset[0]), "formulations must differ by a constant only"
    assert np.argmax(proper) == np.argmax(logit_form)
    assert boundaries is not None


def test_negative_evidence_makes_segmentation_evidence_driven():
    """With a flat prior, boundaries should follow the probabilities, not the length."""
    probs = np.full(30, 0.01)
    probs[9] = 0.99
    probs[19] = 0.99
    flat = create_prior_function("uniform", {})

    without = constrained_segmentation(probs, flat, use_negative_evidence=False)
    with_ne = constrained_segmentation(probs, flat, use_negative_evidence=True)

    # A uniform prior gives the old objective no reason to split at all.
    assert list(without) == []
    # The corrected objective recovers exactly the two confident boundaries.
    assert list(with_ne) == [10, 20]


def test_length_constraints_still_respected():
    rng = np.random.default_rng(7)
    probs = rng.random(120) * 0.6
    prior_fn = gaussian(target=25, spread=10)

    boundaries = constrained_segmentation(
        probs, prior_fn, min_length=10, max_length=40, use_negative_evidence=True
    )

    cuts = [0, *boundaries, len(probs)]
    lengths = [b - a for a, b in zip(cuts, cuts[1:])]
    assert all(length <= 40 for length in lengths), lengths


@pytest.mark.parametrize("algorithm", ["viterbi", "greedy"])
def test_saturated_probabilities_do_not_produce_nan(algorithm):
    """p exactly 0 or 1 must stay finite under the log-odds form."""
    probs = np.array([0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.3])
    prior_fn = gaussian(target=3, spread=2)

    boundaries = constrained_segmentation(
        probs, prior_fn, algorithm=algorithm, use_negative_evidence=True
    )

    assert all(0 < b < len(probs) for b in boundaries)


@pytest.mark.parametrize("algorithm", ["viterbi", "greedy"])
def test_runs_on_degenerate_inputs(algorithm):
    prior_fn = gaussian(target=3, spread=2)
    for probs in (np.array([0.5]), np.array([0.5, 0.5])):
        result = constrained_segmentation(
            probs, prior_fn, algorithm=algorithm, use_negative_evidence=True
        )
        assert isinstance(result, list)
