import numpy as np

def constrained_segmentation(
    probs,
    prior_fn,
    min_length=1,
    max_length=None,
    algorithm="viterbi",
):
    """
    Segments text based on probabilities and length constraints.

    Args:
        probs: Array of probabilities (scores) for each unit.
        prior_fn: Function that takes a length and returns a prior probability.
        min_length: Minimum length of a chunk.
        max_length: Maximum length of a chunk.
        algorithm: "viterbi" or "greedy".

    Returns:
        List of indices where splits occur (end of chunk).
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
                    prev_split = indices[-1] if indices else 0
                    if current_idx - prev_split + remaining <= max_length:
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
                    prev_split = indices[-1] if indices else 0
                    if current_idx - prev_split + remaining <= max_length:
                        indices.pop()
                        return indices
                break

            indices.append(best_end)
            current_idx = best_end

        return indices

    elif algorithm == "viterbi":
        dp = np.full(n + 1, -float("inf"))
        dp[0] = 0.0
        backpointers = np.zeros(n + 1, dtype=int)

        with np.errstate(divide='ignore'):
            log_probs = np.log(probs)

        for i in range(1, n + 1):
            start_j = max(0, i - max_length)
            end_j = i - min_length

            if end_j < start_j:
                continue

            for j in range(start_j, end_j + 1):
                length = i - j
                prior = prior_fn(length)
                if prior <= 0:
                    continue

                log_prior = np.log(prior)
                current_score = dp[j] + log_prior

                if i < n:
                    current_score += log_probs[i-1]

                if current_score > dp[i]:
                    dp[i] = current_score
                    backpointers[i] = j

        indices = []
        curr = n

        if dp[n] == -float("inf"):
            curr_idx = 0
            while curr_idx < n:
                next_split = min(curr_idx + max_length, n)
                if next_split > curr_idx + min_length:
                    indices.append(next_split)
                curr_idx = next_split

            if indices and n - indices[-1] < min_length:
                if len(indices) > 1:
                    prev_split = indices[-2]
                    if n - prev_split <= max_length:
                        indices.pop()
                    else:
                        indices[-1] = max(indices[-1], n - max_length)
                elif n <= max_length:
                    return []
            return indices

        while curr > 0:
            prev = backpointers[curr]
            if prev == 0 and curr != n:
                indices = []
                curr_idx = 0
                while curr_idx < n:
                    next_split = min(curr_idx + max_length, n)
                    if next_split > curr_idx + min_length:
                        indices.append(next_split)
                    curr_idx = next_split

                if indices and n - indices[-1] < min_length:
                    if len(indices) > 1:
                        prev_split = indices[-2]
                        if n - prev_split <= max_length:
                            indices.pop()
                        else:
                            indices[-1] = max(indices[-1], n - max_length)
                    elif n <= max_length:
                        return []
                return indices
            indices.append(curr)
            curr = prev

        result = indices[::-1]

        if result and result[-1] == n:
            result = result[:-1]

        if result:
            last_chunk_len = n - result[-1]
            if last_chunk_len < min_length:
                if len(result) > 1:
                    prev_split = result[-2]
                    if n - prev_split <= max_length:
                        result.pop()
                    else:
                        if result[-1] - result[-2] <= max_length:
                            result[-1] = max(result[-1], n - max_length)
                else:
                    if n <= max_length:
                        return []

        return result

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
