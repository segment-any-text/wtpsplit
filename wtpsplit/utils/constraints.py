import numpy as np

from wtpsplit.utils import indices_to_sentences


def _enforce_segment_constraints(text, indices, min_length, max_length, strip_whitespace=False):
    """
    Extract segments from text using indices, enforcing STRICT length constraints.

    NOTE: This post-processing is necessary because Viterbi operates on raw character
    indices, but text extraction extends segments to include trailing whitespace (which
    can exceed max_length) and optionally strips whitespace (which can go below min_length).

    Guarantees:
    - All segments are strictly <= max_length characters
    - All segments are >= min_length characters (best effort)
    - "".join(segments) == original_text (text preservation)

    Args:
        text: Original text string
        indices: List of split indices (positions where segments end)
        min_length: Minimum segment length
        max_length: Maximum segment length (None for no limit)
        strip_whitespace: Whether to strip whitespace from final segments

    Returns:
        List of segments that respect the constraints
    """
    if not text:
        return []

    # For whitespace-only text, return empty if strip_whitespace, otherwise preserve
    if not text.strip():
        if strip_whitespace:
            return []
        # Text is whitespace-only but we need to preserve it
        if max_length is not None and len(text) > max_length:
            return [text[i : i + max_length] for i in range(0, len(text), max_length)]
        return [text]

    # No constraints - use standard extraction
    if min_length <= 1 and max_length is None:
        return indices_to_sentences(text, indices, strip_whitespace=strip_whitespace)

    # Build initial segment boundaries from indices
    boundaries = []
    offset = 0
    for idx in indices:
        end = idx + 1
        # Extend to include trailing whitespace
        while end < len(text) and text[end].isspace():
            end += 1
        if end > offset:
            boundaries.append((offset, end))
        offset = end
    # Add final segment
    if offset < len(text):
        boundaries.append((offset, len(text)))

    if not boundaries:
        seg = text.strip() if strip_whitespace else text
        if max_length is not None and len(seg) > max_length:
            # Force split - only filter whitespace-only chunks if strip_whitespace is True
            chunks = [seg[i : i + max_length] for i in range(0, len(seg), max_length)]
            if strip_whitespace:
                chunks = [c for c in chunks if c.strip()]
            else:
                chunks = [c for c in chunks if c]  # Only filter truly empty strings
            return chunks
        return [seg] if seg else []

    # Process boundaries to enforce strict max_length while preserving text
    result = []
    pending_prefix = ""  # Whitespace to prepend to next segment
    i = 0

    while i < len(boundaries):
        start, end = boundaries[i]
        segment = pending_prefix + text[start:end]
        pending_prefix = ""

        # STRICT max_length enforcement
        if max_length is not None and len(segment) > max_length:
            # Split this segment to fit max_length
            while len(segment) > max_length:
                # Find a good split point (prefer splitting at whitespace)
                split_at = max_length
                # Look for whitespace near the end to split at
                for j in range(max_length - 1, max(0, max_length - 20), -1):
                    if segment[j].isspace():
                        split_at = j + 1
                        break

                chunk = segment[:split_at]
                segment = segment[split_at:]

                if strip_whitespace:
                    chunk = chunk.strip()
                if chunk:
                    result.append(chunk)

            # Handle remaining part
            if segment:
                # Check if remaining can be merged with next segment
                if i + 1 < len(boundaries):
                    pending_prefix = segment
                else:
                    if strip_whitespace:
                        segment = segment.strip()
                    if segment:
                        result.append(segment)
            i += 1
            continue

        # Check min_length - merge with next if too short
        seg_len = len(segment.strip()) if strip_whitespace else len(segment)
        if seg_len < min_length and i + 1 < len(boundaries):
            # Try to merge with next segment
            j = i + 1

            while j < len(boundaries) and seg_len < min_length:
                _, next_end = boundaries[j]
                # Merge by appending text from current end to next boundary
                merged = segment + text[end:next_end] if segment else text[start:next_end]
                merged_len = len(merged.strip()) if strip_whitespace else len(merged)

                # Check strict max_length
                if max_length is not None and merged_len > max_length:
                    break

                segment = merged
                end = next_end  # Update end to track where we've merged up to
                seg_len = merged_len
                j += 1

            if strip_whitespace:
                segment = segment.strip()
            if segment:
                result.append(segment)
            i = j
            pending_prefix = ""
            continue

        # Segment is valid
        if strip_whitespace:
            segment = segment.strip()
        if segment:
            result.append(segment)
        i += 1

    # Handle any remaining prefix
    if pending_prefix:
        if result:
            # Try to append to last segment
            last = result[-1]
            merged = last + pending_prefix
            if max_length is None or len(merged) <= max_length:
                result[-1] = merged
            else:
                result.append(pending_prefix.strip() if strip_whitespace else pending_prefix)
        else:
            result.append(pending_prefix.strip() if strip_whitespace else pending_prefix)

    # Final cleanup: merge last segment if too short
    if len(result) > 1:
        last = result[-1]
        last_len = len(last.strip()) if strip_whitespace else len(last)
        if last_len < min_length:
            prev = result[-2]
            merged = prev + last
            if max_length is None or len(merged) <= max_length:
                result[-2] = merged
                result.pop()

    # Return all segments to preserve text (don't filter whitespace-only)
    return result


def _handle_short_final_segment(indices, n, min_length, max_length):
    """
    Handle final segment that is too short by merging or adjusting split points.

    This helper eliminates duplication between fallback and post-processing logic.

    Args:
        indices: List of split positions (modified in-place if needed)
        n: Total length
        min_length: Minimum segment length
        max_length: Maximum segment length

    Returns:
        Modified indices list
    """
    if not indices:
        return indices

    last_chunk_len = n - indices[-1]
    if last_chunk_len >= min_length:
        return indices

    # Final segment is too short
    if len(indices) > 1:
        prev_split = indices[-2]
        # Try to merge with previous: remove last split if result fits max_length
        if n - prev_split <= max_length:
            indices.pop()
        else:
            # Can't merge - try to move split point to satisfy min_length
            desired_split = n - min_length
            min_valid_split = prev_split + 1
            adjusted_split = max(desired_split, min_valid_split)
            # Ensure previous chunk doesn't exceed max_length
            if adjusted_split - prev_split <= max_length:
                indices[-1] = adjusted_split
    else:
        # Single split - try to adjust or remove
        if n <= max_length:
            return []
        else:
            # Try to move split to satisfy min_length for final chunk
            desired_split = n - min_length
            if desired_split >= min_length:  # first chunk also needs min_length
                indices[-1] = desired_split

    return indices


def _fallback_greedy_segmentation(n, min_length, max_length):
    """
    Generate greedy segmentation when DP fails (fallback).

    Used when Viterbi DP table cannot reach the end position.

    Args:
        n: Total length
        min_length: Minimum segment length
        max_length: Maximum segment length

    Returns:
        List of split positions
    """
    indices = []
    curr_idx = 0

    while curr_idx < n:
        next_split = min(curr_idx + max_length, n)
        # Use >= to handle min_length == max_length case
        if next_split >= curr_idx + min_length:
            indices.append(next_split)
        curr_idx = next_split

    return _handle_short_final_segment(indices, n, min_length, max_length)


def constrained_segmentation(
    probs,
    prior_fn,
    min_length=1,
    max_length=None,
    algorithm="viterbi",
):
    """
    Segment text with explicit length constraints using dynamic programming.

    The optimization objective for split positions C = {c1, ..., ck} is:

        argmax_C  sum_i [ log prior(ci - c{i-1}) + log p(ci) ]

    where c0 = 0 and p(ci) is omitted for the terminal boundary ci = n
    (there is no split probability at end-of-text).

    Viterbi state definition:
        dp[i] = best log-score for segmenting prefix [0:i]
    Transition:
        dp[i] = max_j dp[j] + log prior(i-j) + log probs[i-1]   (i < n)
        dp[n] = max_j dp[j] + log prior(n-j)
    with j constrained so each segment length (i-j) satisfies
    min_length <= (i-j) <= max_length.

    This function returns boundary positions as 1-based end indices in [1, n).
    Callers convert to 0-based split indices for text extraction.

    Args:
        probs: Array of probabilities (scores) for each unit.
        prior_fn: Function that takes a length and returns a prior probability.
        min_length: Minimum length of a chunk.
        max_length: Maximum length of a chunk.
        algorithm: "viterbi" or "greedy".

    Returns:
        List[int]: split boundary end-positions (excluding n).
    """
    n = len(probs)
    if max_length is None:
        max_length = n

    if algorithm == "greedy":
        # Simple greedy approach (not optimal)
        indices = []
        current_idx = 0
        while current_idx < n:
            best_score = -float("inf")
            best_end = -1

            start_search = current_idx + min_length
            end_search = min(current_idx + max_length + 1, n + 1)

            if start_search >= end_search:
                remaining = n - current_idx
                if remaining < min_length and indices:
                    # Want to merge remaining with previous chunk by removing last split
                    # But must verify the resulting final segment fits in max_length
                    new_last_split = indices[-2] if len(indices) >= 2 else 0
                    if n - new_last_split <= max_length:
                        indices.pop()
                        return indices
                best_end = n
            else:
                for end in range(start_search, end_search):
                    if end == n:
                        score = prior_fn(end - current_idx)
                    else:
                        score = probs[end - 1] * prior_fn(end - current_idx)

                    if score > best_score:
                        best_score = score
                        best_end = end

                if best_end == -1:
                    best_end = min(current_idx + max_length, n)

            if best_end == n:
                remaining = n - current_idx
                if remaining < min_length and indices:
                    # Want to merge remaining with previous chunk by removing last split
                    # But must verify the resulting final segment fits in max_length
                    new_last_split = indices[-2] if len(indices) >= 2 else 0
                    if n - new_last_split <= max_length:
                        indices.pop()
                        return indices
                break

            indices.append(best_end)
            current_idx = best_end

        return indices

    elif algorithm == "viterbi":
        # ============================================================================
        # VITERBI DYNAMIC PROGRAMMING ALGORITHM
        # ============================================================================
        # Goal: Find optimal segmentation that maximizes:
        #   Score = ∏ Prior(segment_length) × P(boundary)
        # In log-space (to prevent underflow):
        #   Log-Score = ∑ log(Prior(length)) + ∑ log(P(boundary))
        #
        # dp[current_pos] = best log-score to segment text[0:current_pos]
        # backpointers[current_pos] = where the last segment started (for reconstruction)
        # ============================================================================

        # Initialize DP table: all positions unreachable except start
        dp = np.full(n + 1, -float("inf"))
        dp[0] = 0.0  # Base case: no text segmented = score of 1, log(1) = 0
        backpointers = np.zeros(n + 1, dtype=int)

        # Convert boundary probabilities to log-space for numerical stability
        # log(a × b) = log(a) + log(b) prevents underflow from multiplying small numbers
        with np.errstate(divide="ignore"):
            log_probs = np.log(probs)

        # Fill DP table: for each position (potential segment endpoint)
        for current_pos in range(1, n + 1):
            # ========================================================================
            # Find valid segment start positions that satisfy length constraints
            # ========================================================================
            # Segment from segment_start to current_pos has length (current_pos - segment_start)
            # Must satisfy: min_length ≤ segment_length ≤ max_length
            #
            # Rearranging:
            #   segment_length ≤ max_length
            #   → current_pos - segment_start ≤ max_length
            #   → segment_start ≥ current_pos - max_length
            #   → earliest_start = max(0, current_pos - max_length)
            #
            #   segment_length ≥ min_length
            #   → current_pos - segment_start ≥ min_length
            #   → segment_start ≤ current_pos - min_length
            #   → latest_start = current_pos - min_length
            # ========================================================================

            earliest_start = max(0, current_pos - max_length)  # Can't exceed max_length
            latest_start = current_pos - min_length  # Must meet min_length

            # If no valid segment lengths exist (e.g., current_pos=5, min_length=10), skip
            if latest_start < earliest_start:
                continue

            # Try all valid segment start positions
            for segment_start in range(earliest_start, latest_start + 1):
                segment_length = current_pos - segment_start

                # Check if this segment length is allowed by the prior distribution
                prior_probability = prior_fn(segment_length)
                if prior_probability <= 0:
                    continue  # This length is forbidden (zero probability)

                log_prior = np.log(prior_probability)

                # ====================================================================
                # Calculate score for this segmentation choice
                # ====================================================================
                # Score = (best score to reach segment_start)
                #       + (log-prior for this segment length)
                #       + (log-probability of boundary at current_pos, if not at end)
                # ====================================================================
                candidate_score = dp[segment_start] + log_prior

                # Add boundary probability only for real boundaries (not the final position)
                if current_pos < n:
                    # Model's prediction for splitting at this position
                    # probs[0] = boundary after char 0, so probs[current_pos-1] = boundary at current_pos
                    boundary_prob_index = current_pos - 1
                    candidate_score += log_probs[boundary_prob_index]
                # Note: At current_pos == n (end of text), no boundary probability is added
                # because the end is always a boundary (no model prediction needed)

                # Update DP table if this is the best way to reach current_pos
                if candidate_score > dp[current_pos]:
                    dp[current_pos] = candidate_score
                    backpointers[current_pos] = segment_start

        # Handle DP failure: if we can't reach the end, use greedy fallback
        if dp[n] == -float("inf"):
            return _fallback_greedy_segmentation(n, min_length, max_length)

        # Reconstruct path from backpointers
        indices = []
        current_pos = n
        while current_pos > 0:
            prev_pos = backpointers[current_pos]
            indices.append(current_pos)
            current_pos = prev_pos

        # Reverse to get forward order
        result = indices[::-1]

        # Remove terminal boundary (n is always a boundary, not a split)
        if result and result[-1] == n:
            result = result[:-1]

        # Handle short final segment
        return _handle_short_final_segment(result, n, min_length, max_length)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
