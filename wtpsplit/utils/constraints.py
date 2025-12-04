import numpy as np

from wtpsplit.utils import indices_to_sentences


def _enforce_segment_constraints(text, indices, min_length, max_length, strip_whitespace=False):
    """
    Extract segments from text using indices, enforcing STRICT length constraints.
    
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
            return [text[i:i+max_length] for i in range(0, len(text), max_length)]
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
            chunks = [seg[i:i+max_length] for i in range(0, len(seg), max_length)]
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


def _enforce_segment_constraints_simple(sentences, min_length, max_length, delimiter=" "):
    """
    Simple constraint enforcement for already-extracted segments.
    
    Used when we know the delimiter between segments (e.g., after split("\\n")).
    Merges segments using the specified delimiter to preserve text structure.
    
    Guarantees:
    - All segments are strictly <= max_length characters (STRICT)
    - All segments are >= min_length characters (BEST EFFORT - may not be achievable
      if merging would violate max_length or if segment is inherently too short)
    - delimiter.join(segments) preserves text structure
    
    Args:
        sentences: List of text segments
        min_length: Minimum segment length (best effort)
        max_length: Maximum segment length (None for no limit, strictly enforced)
        delimiter: Delimiter to use when merging segments
    
    Returns:
        List of segments with constraints applied (max_length always respected,
        min_length satisfied where possible)
    """
    if not sentences:
        return sentences
    
    # Preserve structure: keep empty strings (they represent consecutive delimiters/newlines)
    # This matches baseline behavior when split_on_input_newlines=True
    if min_length <= 1 and max_length is None:
        return sentences
    
    # Process only non-empty segments, but track indices to preserve empty ones
    result = []
    i = 0
    
    while i < len(sentences):
        seg = sentences[i]
        
        # Preserve empty strings (consecutive delimiter markers)
        if not seg or not seg.strip():
            result.append(seg)
            i += 1
            continue
        
        seg_len = len(seg)
        
        # STRICT max_length enforcement - split if too long
        if max_length is not None and seg_len > max_length:
            for offset in range(0, seg_len, max_length):
                chunk = seg[offset:offset + max_length]
                if chunk:
                    result.append(chunk)
            i += 1
            continue
        
        # Segment too short - merge with next non-empty using delimiter
        if seg_len < min_length:
            merged = seg
            j = i + 1
            pending_delimiters = ""  # Track delimiters from empty segments
            trailing_empty = []  # Track empty segments after last successful merge
            
            while j < len(sentences) and len(merged) < min_length:
                next_seg = sentences[j]
                
                # Empty segments represent consecutive delimiters - accumulate them
                if not next_seg or not next_seg.strip():
                    pending_delimiters += delimiter  # Each empty = one more delimiter
                    trailing_empty.append(next_seg)
                    j += 1
                    continue
                
                # Build merged string: base delimiter + any accumulated from empty segments
                all_delims = delimiter + pending_delimiters
                new_merged = merged + all_delims + next_seg
                
                # STRICT max_length check
                if max_length is not None and len(new_merged) > max_length:
                    break
                
                merged = new_merged
                pending_delimiters = ""  # Reset after successful merge
                trailing_empty = []  # Clear since they're absorbed
                j += 1
            
            # If still too short, try merging with previous non-empty segment
            if len(merged) < min_length and result:
                # Find a previous non-empty segment that can accommodate the merge
                for prev_idx in range(len(result) - 1, -1, -1):
                    if result[prev_idx] and result[prev_idx].strip():
                        # Count empty segments between prev_idx and end of result
                        # They represent consecutive delimiters that must be preserved
                        empty_between = len(result) - prev_idx - 1
                        all_delims = delimiter * (empty_between + 1)
                        prev_merged = result[prev_idx] + all_delims + merged
                        if max_length is None or len(prev_merged) <= max_length:
                            result[prev_idx] = prev_merged
                            # Remove all segments after prev_idx (they're now in the merged string)
                            del result[prev_idx + 1:]
                            merged = None  # Mark as merged into previous
                            break
                        # If this one doesn't fit, try earlier segments
            
            if merged is not None:
                result.append(merged)
            
            # Preserve any trailing empty segments that weren't absorbed
            result.extend(trailing_empty)
            
            i = j
            continue
        
        result.append(seg)
        i += 1
    
    # Merge last non-empty segment if too short
    non_empty_indices = [i for i, s in enumerate(result) if s and s.strip()]
    if len(non_empty_indices) >= 2:
        last_idx = non_empty_indices[-1]
        prev_idx = non_empty_indices[-2]
        last = result[last_idx]
        prev = result[prev_idx]
        
        if len(last) < min_length:
            # Count empty segments between prev and last (they represent delimiters)
            empty_count = last_idx - prev_idx - 1
            # Build merge with all intermediate delimiters: prev + (empty_count + 1) delimiters + last
            all_delims = delimiter * (empty_count + 1)
            merged = prev + all_delims + last
            # STRICT max_length check
            if max_length is None or len(merged) <= max_length:
                result[prev_idx] = merged
                # Remove all segments from prev_idx+1 to last_idx (inclusive)
                del result[prev_idx + 1 : last_idx + 1]
    
    return result


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
                # Use >= to handle min_length == max_length case
                if next_split >= curr_idx + min_length:
                    indices.append(next_split)
                curr_idx = next_split

            if indices and n - indices[-1] < min_length:
                if len(indices) > 1:
                    prev_split = indices[-2]
                    # Try to merge with previous: remove last split if result fits max_length
                    if n - prev_split <= max_length:
                        indices.pop()
                    else:
                        # Can't merge - try to move split point to satisfy min_length
                        # New split should give final chunk >= min_length
                        desired_split = n - min_length
                        # But previous chunk must stay <= max_length
                        min_valid_split = prev_split + 1  # at least 1 char in prev chunk after prev_split
                        # And previous chunk must stay >= min_length (best effort)
                        adjusted_split = max(desired_split, min_valid_split)
                        # Ensure we don't exceed max_length for previous chunk
                        if adjusted_split - prev_split <= max_length:
                            indices[-1] = adjusted_split
                        # else: keep current split (best effort - one constraint must give)
                elif n <= max_length:
                    # Single split that leaves short final - just remove it
                    return []
            return indices

        while curr > 0:
            prev = backpointers[curr]
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
                    # Try to merge with previous: remove last split if result fits max_length
                    if n - prev_split <= max_length:
                        result.pop()
                    else:
                        # Can't merge - try to move split point to satisfy min_length
                        desired_split = n - min_length
                        min_valid_split = prev_split + 1
                        adjusted_split = max(desired_split, min_valid_split)
                        # Ensure previous chunk doesn't exceed max_length
                        if adjusted_split - prev_split <= max_length:
                            result[-1] = adjusted_split
                        # else: keep current split (best effort)
                else:
                    # Single split - try to adjust or remove
                    if n <= max_length:
                        return []
                    else:
                        # Try to move split to satisfy min_length for final chunk
                        desired_split = n - min_length
                        if desired_split >= min_length:  # first chunk also needs min_length
                            result[-1] = desired_split

        return result

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
