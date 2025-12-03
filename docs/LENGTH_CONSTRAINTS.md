# Length-Constrained Segmentation

This supplementary document explains the theory and implementation of length-constrained segmentation in wtpsplit (NB: auto-generated).

## Overview

A text segmenter like SaT gives us a probability score for every character, indicating how likely that position is to be a segment boundary (e.g., end of a sentence).

### The Basic Approach

Simply split whenever `probability > threshold`.

**Problem**: No control over segment lengths!

### The Controllable Approach

> **Note**: When using length-constrained segmentation (`max_length` is set), the `threshold` parameter is **ignored**. The algorithms use raw model probabilities directly to find optimal split points.

Define a **prior probability distribution** over chunk lengths, then solve an optimization problem that balances:
- The model's boundary predictions
- Your length preferences

## The Mathematics

We view segmentation as selecting a subset of positions C = {C₁, C₂, ..., Cₖ} where each Cᵢ marks the end of a chunk.

**Optimization Problem:**

```
argmax_C  ∏ᵢ Prior(Cᵢ - Cᵢ₋₁) × P(Cᵢ)
```

Where:
- `Cᵢ` = position of the i-th split point
- `Prior(length)` = how much we prefer chunks of that length  
- `P(Cᵢ)` = model's probability at position Cᵢ

In Bayesian terms:
- `Prior(Cᵢ - Cᵢ₋₁)` is the prior
- `P(Cᵢ)` is the evidence
- Their product is the posterior

## Prior Functions

### 1. Uniform Prior

```python
Prior(length) = 1.0 if length ≤ max_length else 0.0
```

- All lengths equally good up to `max_length`
- Hard cutoff at maximum
- **Use case**: Simple length limiting

### 2. Gaussian Prior

```python
Prior(length) = exp(-0.5 × ((length - μ) / σ)²)
```

- Peaks at `μ` (preferred length)
- Falls off smoothly based on `σ`
- **Use case**: Prefer specific chunk sizes (e.g., for embeddings)

### 3. Clipped Polynomial Prior

```python
Prior(length) = max(1 - α × (length - μ)², 0)
```

- Peaks at `μ`
- Quadratic falloff controlled by `α`
- Clips to zero far from peak
- **Use case**: Strong preference for target length

## Algorithms

### Greedy Search

At each step, pick the locally best split point based on `Prior(length) × P(position)`.

**Pros:**
- Fast (O(n × max_length))
- Simple to understand

**Cons:**
- Not globally optimal
- May miss better overall segmentations

### Viterbi Algorithm (Recommended)

Dynamic programming to find the globally optimal sequence of splits.

**Algorithm:**
```
dp[i] = best score achievable for text[0:i]
dp[i] = max over j of: dp[j] + log(Prior(i-j)) + log(P[i-1])

Where j ranges from max(0, i-max_length) to i-min_length
```

**Pros:**
- Globally optimal solution
- Respects sentence boundaries when possible

**Cons:**
- Slightly slower (O(n × max_length))

## Key Guarantees

1. **`max_length` is STRICT**: No segment will ever exceed `max_length` characters
2. **`min_length` is BEST EFFORT**: Segments may be shorter if merging would violate `max_length`
3. **Text preservation**:
   - Default (`split_on_input_newlines=False` or WtP): `"".join(segments) == original_text`
   - With `split_on_input_newlines=True` (SaT default): `"\n".join(segments) == original_text`

## Usage Examples

### Basic Length Limiting

```python
from wtpsplit import SaT

sat = SaT("sat-3l-sm")

# Limit segments to 100 characters
segments = sat.split(text, max_length=100)
```

### Both Min and Max

```python
# Segments between 20-100 characters
segments = sat.split(text, min_length=20, max_length=100)
```

### Using Gaussian Prior

```python
# Prefer ~50 character segments
segments = sat.split(
    text,
    max_length=100,
    prior_type="gaussian",
    prior_kwargs={"mu": 50, "sigma": 15}
)
```

### Algorithm Selection

```python
# Use greedy for speed (slightly suboptimal)
segments = sat.split(text, max_length=100, algorithm="greedy")

# Use viterbi for optimal results (default)
segments = sat.split(text, max_length=100, algorithm="viterbi")
```

## How It Respects Sentence Boundaries

The Viterbi algorithm naturally prefers splitting at high-probability positions (sentence boundaries) because:

1. `P(position)` is high at sentence boundaries
2. The product `Prior(length) × P(position)` is maximized when both factors are high
3. The algorithm finds the global optimum, so it won't make a locally good choice that leads to bad splits later

**Example:**

Text: `"The quick brown fox jumps. Pack my box with jugs."`

With `max_length=100`:
- Position 25 (after "jumps.") has P ≈ 0.95
- Position 50 (after "jugs.") has P ≈ 0.98
- The algorithm will split at these natural boundaries

With `max_length=30`:
- Must split somewhere before position 30
- Will choose position 25 (sentence boundary) over position 28 (mid-word)

## When Word Cuts Happen

Word cuts only occur when there is **no sentence boundary within `max_length`** characters. This happens with:

1. Very long sentences without punctuation
2. Very restrictive `max_length` values
3. Text without natural break points (e.g., code, URLs)

## Implementation Details

### Post-Processing (`_enforce_segment_constraints`)

After the algorithm finds split points, post-processing ensures:
- Segments don't exceed `max_length` (force-splits if needed)
- Short segments are merged when possible
- All whitespace is preserved

### Simple Constraints (`_enforce_segment_constraints_simple`)

Used after `split("\n")` to re-apply constraints:
- Merges short segments with delimiter
- Preserves empty strings (consecutive newlines)
- Best-effort `min_length`, strict `max_length`

## Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_length` | int | 1 | Minimum segment length (best effort) |
| `max_length` | int | None | Maximum segment length (strict) |
| `algorithm` | str | "viterbi" | "viterbi" (optimal) or "greedy" (faster) |
| `prior_type` | str | "uniform" | "uniform", "gaussian", or "clipped_polynomial" |
| `prior_kwargs` | dict | None | Parameters for prior function |

### Prior Parameters

**Gaussian:**
- `mu`: Mean (preferred length), default 20.0
- `sigma`: Standard deviation, default 5.0

**Clipped Polynomial:**
- `mu`: Peak position, default 3.0
- `alpha`: Falloff rate, default 0.5

## See Also

- [Interactive Demo](../length_constrained_segmentation_demo.py) - Run examples and experiment
- [Test Suite](../test_length_constraints.py) - Comprehensive tests
- [README](../README.md) - Quick start guide

