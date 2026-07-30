"""Compact exact and conservative near-duplicate evaluation index."""

from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
import re
import unicodedata

SIMHASH_BANDS = 4
SIMHASH_BITS_PER_BAND = 16
NEAR_HAMMING_THRESHOLD = 12
MIN_NEAR_CHARS = 40


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(text.split())


def text_hash(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def candidate_units(text: str) -> list[str]:
    units = [text]
    units.extend(
        part
        for part in re.split(r"(?:\r?\n)+|(?<=[.!?。！？])\s+", text)
        if part.strip()
    )
    return list(dict.fromkeys(normalize_text(unit) for unit in units if unit.strip()))


def shingle_hashes(normalized: str, width: int = 13) -> list[int]:
    compact = normalized.replace(" ", "")
    if len(compact) <= width:
        values = [compact]
    else:
        values = [compact[index : index + width] for index in range(0, len(compact) - width + 1, 3)]
    return [
        int.from_bytes(
            hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        for value in values
        if value
    ]


def simhash(normalized: str) -> int:
    weights = [0] * 64
    for value in shingle_hashes(normalized):
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    output = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            output |= 1 << bit
    return output


def band_keys(value: int) -> list[str]:
    mask = (1 << SIMHASH_BITS_PER_BAND) - 1
    return [
        f"{band}:{(value >> (band * SIMHASH_BITS_PER_BAND)) & mask}"
        for band in range(SIMHASH_BANDS)
    ]


class ReferenceIndex:
    def __init__(self) -> None:
        self.exact_keys: set[str] = set()
        self.near_entries: list[dict] = []
        self._bands: dict[str, list[int]] = defaultdict(list)
        self.source_counts: dict[str, int] = defaultdict(int)

    @staticmethod
    def language_key(language_script: str | None) -> str:
        return language_script.split("_", 1)[0] if language_script else "*"

    def add_text(
        self,
        text: str,
        reference_id: str,
        source: str,
        language_script: str | None = None,
    ) -> None:
        language = self.language_key(language_script)
        for unit_number, normalized in enumerate(candidate_units(text)):
            if not normalized:
                continue
            self.exact_keys.add(f"{language}\0{text_hash(normalized)}")
            self.source_counts[source] += 1
            if len(normalized) < MIN_NEAR_CHARS:
                continue
            value = simhash(normalized)
            entry_number = len(self.near_entries)
            self.near_entries.append(
                {
                    "reference_id": reference_id,
                    "unit_number": unit_number,
                    "source": source,
                    "language": language,
                    "simhash": value,
                }
            )
            for key in band_keys(value):
                self._bands[key].append(entry_number)

    def rebuild_bands(self) -> None:
        self._bands = defaultdict(list)
        for index, entry in enumerate(self.near_entries):
            for key in band_keys(entry["simhash"]):
                self._bands[key].append(index)

    def match(self, text: str, language_script: str | None = None) -> dict | None:
        language = self.language_key(language_script)
        normalized_units = candidate_units(text)
        for unit in normalized_units:
            digest = text_hash(unit)
            if (
                f"{language}\0{digest}" in self.exact_keys
                or f"*\0{digest}" in self.exact_keys
                or (language == "*" and any(key.endswith(f"\0{digest}") for key in self.exact_keys))
            ):
                return {"match": "exact", "unit_sha256": digest}
        best = None
        for unit in normalized_units:
            if len(unit) < MIN_NEAR_CHARS:
                continue
            value = simhash(unit)
            candidate_indices = {
                index
                for key in band_keys(value)
                for index in self._bands.get(key, ())
            }
            for index in candidate_indices:
                entry = self.near_entries[index]
                if language != "*" and entry.get("language", "*") != language:
                    continue
                distance = (value ^ entry["simhash"]).bit_count()
                if distance <= NEAR_HAMMING_THRESHOLD and (
                    best is None or distance < best["hamming_distance"]
                ):
                    best = {
                        "match": "near",
                        "reference_id": entry["reference_id"],
                        "source": entry["source"],
                        "hamming_distance": distance,
                    }
        return best

    def to_payload(self) -> dict:
        return {
            "version": "mmsat_contamination_index_v2",
            "algorithm": {
                "normalization": "Unicode NFKC, casefold, collapsed whitespace",
                "exact": (
                    "SHA-256 of documents and coarse sentence/line units, keyed "
                    "by ISO language"
                ),
                "near": (
                    f"64-bit character-shingle SimHash, {SIMHASH_BANDS}x"
                    f"{SIMHASH_BITS_PER_BAND}-bit LSH, Hamming <= "
                    f"{NEAR_HAMMING_THRESHOLD}; candidates restricted to the "
                    "same ISO language"
                ),
                "near_policy": "Conservative quarantine; never silently retain.",
            },
            "counts": {
                "exact_hashes": len(self.exact_keys),
                "near_signatures": len(self.near_entries),
                "by_source": dict(self.source_counts),
            },
            "exact_keys": sorted(self.exact_keys),
            "near_entries": self.near_entries,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "ReferenceIndex":
        index = cls()
        if "exact_keys" in payload:
            index.exact_keys = set(payload["exact_keys"])
        else:
            index.exact_keys = {f"*\0{value}" for value in payload["exact_hashes"]}
        index.near_entries = payload["near_entries"]
        index.source_counts.update(payload.get("counts", {}).get("by_source", {}))
        index.rebuild_bands()
        return index


def load_index(path) -> ReferenceIndex:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, mode="rt", encoding="utf-8") as handle:
        return ReferenceIndex.from_payload(json.load(handle))
