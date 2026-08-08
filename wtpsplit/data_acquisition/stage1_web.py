"""Stage-1 FineWeb / mC4 corpus building.

Resolves Hub sources, applies paragraph or document unit transforms, filters
contamination, and writes resumable per-language parquet shards. Callers use
``scripts/build_stage1.py``; tests may inject a custom shard writer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
import unicodedata


REQUIRED_COLUMNS = ("text", "ends_with_punctuation", "lang")
_PUNCTUATION_CATEGORIES = frozenset({"P", "S"})


@dataclass(frozen=True)
class SourceSpec:
    """Resolved source for one SaT language."""

    lang: str
    corpus: str
    dataset: str
    config: str
    split: str = "train"
    language_script: str | None = None
    revision: str | None = None
    contamination_key: str | None = None


@dataclass(frozen=True)
class BuildOptions:
    """Options whose canonical representation forms part of the resume key."""

    output_dir: Path
    unit: str
    valid_ratio: float = 0.001
    seed: int = 42
    non_punctuation_sample_ratio: float | None = 0.1
    no_punctuation_languages: tuple[str, ...] = ()
    compression: str = "zstd"
    producer: str = "wtpsplit.data_acquisition.stage1_web"
    producer_version: str = "stage1_web_v1"

    def __post_init__(self) -> None:
        if self.unit not in {"paragraph", "document"}:
            raise ValueError("unit must be 'paragraph' or 'document'")
        if not 0.0 < self.valid_ratio < 1.0:
            raise ValueError("valid_ratio must be in (0, 1)")
        ratio = self.non_punctuation_sample_ratio
        if ratio is not None and not 0.0 <= ratio < 1.0:
            raise ValueError("non_punctuation_sample_ratio must be in [0, 1)")


@dataclass(frozen=True)
class ArtifactInfo:
    rows: int
    bytes: int
    sha256: str


class ShardWriter(Protocol):
    """Writer abstraction used to avoid importing pyarrow in fixture tests."""

    def write(self, path: Path, rows: Sequence[Mapping[str, Any]]) -> ArtifactInfo:
        ...


class ParquetShardWriter:
    """Write Stage-1 parquet shards, importing pyarrow only when used."""

    def __init__(self, compression: str = "zstd") -> None:
        self.compression = compression

    def write(self, path: Path, rows: Sequence[Mapping[str, Any]]) -> ArtifactInfo:
        try:
            import pyarrow as pa  # type: ignore[import-untyped]
            import pyarrow.parquet as pq  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError(
                "Writing Stage-1 parquet shards requires optional dependency "
                "`pyarrow`; install the research dependencies or inject a ShardWriter"
            ) from exc
        schema = pa.schema(
            [
                ("text", pa.string()),
                ("ends_with_punctuation", pa.bool_()),
                ("lang", pa.string()),
            ]
        )
        table = pa.Table.from_pylist(list(rows), schema=schema)
        pq.write_table(table, path, compression=self.compression)
        return _artifact_info(path, len(rows))


class JsonlShardWriter:
    """Dependency-free writer useful for smoke tests (suffix is caller-owned)."""

    def write(self, path: Path, rows: Sequence[Mapping[str, Any]]) -> ArtifactInfo:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
        return _artifact_info(path, len(rows))


DocumentProvider = Callable[[SourceSpec], Iterable[str]]
ContaminationMatcher = Callable[[str, str | None], Any]


def _read_payload(value: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return json.loads(Path(value).read_text(encoding="utf-8"))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_mc4_caps(
    value: str | Path | Mapping[str, Any],
    *,
    split: str = "train",
) -> dict[str, int]:
    """Load mC4 character caps from either the measured manifest or a flat map."""

    payload = _read_payload(value)
    if "splits" in payload:
        payload = payload["splits"][split]["per_lang"]
    caps: dict[str, int] = {}
    for lang, raw in payload.items():
        count = raw.get("chars") if isinstance(raw, Mapping) else raw
        if count is None:
            raise ValueError(f"Character cap for {lang!r} is missing")
        count = int(count)
        if count <= 0:
            raise ValueError(f"Character cap for {lang!r} must be positive")
        caps[str(lang)] = count
    if not caps:
        raise ValueError("mC4 cap manifest contains no languages")
    return caps


def load_language_map(
    value: str | Path | Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    payload = _read_payload(value)
    rows = payload.get("rows", payload)
    if isinstance(rows, Mapping):
        return {str(lang): dict(row) for lang, row in rows.items()}
    return {str(row["sat_lang"]): dict(row) for row in rows}


def load_script_remaps(
    value: str | Path | Mapping[str, Any] | None,
) -> dict[str, str]:
    payload = _read_payload(value)
    rows = payload.get("remaps", ())
    return {
        str(row["sat_lang"]): str(row["to_fineweb2_language_script"])
        for row in rows
    }


def load_plan(
    value: str | Path | Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    payload = _read_payload(value)
    return {
        str(row["language_script"]): dict(row)
        for row in payload.get("rows", ())
    }


def resolve_web_sources(
    *,
    corpus: str,
    language_map: str | Path | Mapping[str, Any],
    languages: Iterable[str] | None = None,
    plan: str | Path | Mapping[str, Any] | None = None,
    script_remaps: str | Path | Mapping[str, Any] | None = None,
    revisions: Mapping[str, str] | None = None,
    english_config: str | None = None,
) -> dict[str, SourceSpec]:
    """Resolve FineWeb/mC4 inputs without opening a network connection.

    Script remaps override the base mapping before the plan lookup.  They are
    therefore normal source resolution for a new build, never an append/top-up.
    """

    if corpus not in {"fineweb2", "mc4"}:
        raise ValueError("corpus must be 'fineweb2' or 'mc4'")
    mapping = load_language_map(language_map)
    wanted = sorted(set(languages) if languages is not None else mapping)
    plan_by_script = load_plan(plan)
    remaps = load_script_remaps(script_remaps)
    revisions = dict(revisions or {})
    resolved: dict[str, SourceSpec] = {}

    for lang in wanted:
        row = mapping.get(lang)
        if row is None:
            raise ValueError(f"No Stage-1 language mapping for {lang!r}")
        if corpus == "mc4":
            config = "iw" if lang == "he" else lang
            resolved[lang] = SourceSpec(
                lang=lang,
                corpus=corpus,
                dataset="allenai/c4",
                config=config,
                revision=revisions.get(lang) or revisions.get("allenai/c4"),
            )
            continue

        english = row.get("fineweb_english")
        if isinstance(english, Mapping):
            dataset = str(english.get("hf_dataset") or "HuggingFaceFW/fineweb")
            config = str(english_config or english.get("hf_config") or "default")
            split = str(english.get("hf_split") or "train")
            script = str(english.get("local_script_id") or "eng_FineWeb")
        else:
            script_value = remaps.get(lang) or row.get(
                "fineweb2_language_script"
            )
            if not script_value:
                raise ValueError(f"No FineWeb source mapping for {lang!r}")
            script = str(script_value)
            plan_row = plan_by_script.get(str(script))
            if plan_row is None:
                raise ValueError(f"No FineWeb plan row for resolved script {script!r}")
            dataset = str(plan_row["hf_dataset"])
            config = str(plan_row["hf_config"])
            split = str(plan_row.get("hf_split") or "train")
        resolved[lang] = SourceSpec(
            lang=lang,
            corpus=corpus,
            dataset=dataset,
            config=config,
            split=split,
            language_script=script,
            revision=revisions.get(lang) or revisions.get(dataset),
            contamination_key=script,
        )
    return resolved


def iter_hub_documents(source: SourceSpec) -> Iterable[str]:
    """Default network provider, with an actionable optional-dependency error."""

    if not source.revision:
        raise ValueError(
            f"Stage-1 source {source.dataset}/{source.config} requires an "
            "immutable Hub revision"
        )
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "Streaming Stage-1 web sources requires optional dependency `datasets`; "
            "install the research dependencies or inject document_provider"
        ) from exc
    stream = load_dataset(
        source.dataset,
        name=source.config,
        split=source.split,
        streaming=True,
        revision=source.revision,
    )
    for sample in stream:
        text = sample.get("text")
        if isinstance(text, str) and text.strip():
            yield text


def ends_with_punctuation(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and unicodedata.category(stripped[-1])[0] in _PUNCTUATION_CATEGORIES


def split_paragraphs(text: str) -> list[str]:
    return [part + "\n" for part in text.splitlines() if part.strip()]


def paragraph_rows(text: str, lang: str) -> list[dict[str, Any]]:
    return [
        {
            "text": paragraph,
            "ends_with_punctuation": ends_with_punctuation(paragraph),
            "lang": lang,
        }
        for paragraph in split_paragraphs(text)
    ]


def apply_punctuation_mixture(
    paragraphs: Sequence[str],
    *,
    ratio: float | None,
    rng: random.Random,
    language_uses_punctuation: bool,
) -> list[str]:
    """Downsample non-punctuation paragraphs while preserving document order."""

    if ratio is None or not language_uses_punctuation or not paragraphs:
        return list(paragraphs)
    punct_indices = [i for i, value in enumerate(paragraphs) if ends_with_punctuation(value)]
    non_indices = [i for i, value in enumerate(paragraphs) if not ends_with_punctuation(value)]
    if not punct_indices:
        return list(paragraphs)
    target_non = int(len(punct_indices) * ratio / (1.0 - ratio))
    chosen_non = (
        set(rng.sample(non_indices, target_non))
        if 0 < target_non < len(non_indices)
        else (set() if target_non <= 0 else set(non_indices))
    )
    keep = set(punct_indices) | chosen_non
    return [value for i, value in enumerate(paragraphs) if i in keep]


def document_row(
    text: str,
    lang: str,
    *,
    ratio: float | None,
    rng: random.Random,
    language_uses_punctuation: bool,
) -> dict[str, Any] | None:
    paragraphs = apply_punctuation_mixture(
        split_paragraphs(text),
        ratio=ratio,
        rng=rng,
        language_uses_punctuation=language_uses_punctuation,
    )
    if not paragraphs:
        return None
    body = "".join(paragraphs)
    return {
        "text": body,
        "ends_with_punctuation": ends_with_punctuation(paragraphs[-1]),
        "lang": lang,
    }


def deterministic_partition(
    rows: Sequence[Mapping[str, Any]],
    *,
    lang: str,
    valid_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hash-partition units, independent of Python's randomized hash seed."""

    ranked: list[tuple[int, dict[str, Any]]] = []
    denominator = 1 << 256
    for index, raw in enumerate(rows):
        row = dict(raw)
        token = f"{seed}\0{lang}\0{index}\0{row['text']}".encode("utf-8")
        ranked.append((int.from_bytes(hashlib.sha256(token).digest(), "big"), row))
    train = [row for score, row in ranked if score / denominator >= valid_ratio]
    valid = [row for score, row in ranked if score / denominator < valid_ratio]
    # Both artifacts must be non-empty whenever the source provides enough units.
    if len(ranked) >= 2 and not valid:
        selected = min(ranked, key=lambda item: item[0])[1]
        train.remove(selected)
        valid.append(selected)
    elif len(ranked) >= 2 and not train:
        selected = max(ranked, key=lambda item: item[0])[1]
        valid.remove(selected)
        train.append(selected)
    return train, valid


def _canonical(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_canonical(item) for item in value]
    return value


def build_fingerprint(
    *,
    caps: Mapping[str, int],
    sources: Mapping[str, SourceSpec],
    options: BuildOptions,
    input_hashes: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, Any]]:
    params = asdict(options)
    params.pop("output_dir")
    payload = {
        "producer_params": _canonical(params),
        "caps": _canonical(caps),
        "sources": {lang: _canonical(asdict(spec)) for lang, spec in sorted(sources.items())},
        "input_hashes": _canonical(input_hashes or {}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _artifact_info(path: Path, rows: int) -> ArtifactInfo:
    return ArtifactInfo(rows=rows, bytes=path.stat().st_size, sha256=file_sha256(path))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _completed_artifacts_match(
    entry: Mapping[str, Any],
    train_path: Path,
    valid_path: Path,
) -> bool:
    if entry.get("status") != "complete":
        return False
    artifacts = entry.get("artifacts", {})
    for split, path in (("train", train_path), ("valid", valid_path)):
        expected = artifacts.get(split, {})
        if not path.is_file() or expected.get("sha256") != file_sha256(path):
            return False
    return True


def _language_rng(seed: int, lang: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}\0{lang}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def build_stage1_web(
    *,
    caps: Mapping[str, int],
    sources: Mapping[str, SourceSpec],
    options: BuildOptions,
    document_provider: DocumentProvider = iter_hub_documents,
    writer: ShardWriter | None = None,
    contamination_matcher: ContaminationMatcher | None = None,
    input_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build resumable, per-language Stage-1 train/valid shards."""

    normalized_caps = {str(lang): int(value) for lang, value in caps.items()}
    if set(normalized_caps) != set(sources):
        raise ValueError("caps and sources must contain exactly the same languages")
    if any(value <= 0 for value in normalized_caps.values()):
        raise ValueError("all character caps must be positive")

    output_dir = options.output_dir
    train_dir, valid_dir = output_dir / "train", output_dir / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    fingerprint, fingerprint_payload = build_fingerprint(
        caps=normalized_caps,
        sources=sources,
        options=options,
        input_hashes=input_hashes,
    )
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("build_fingerprint") != fingerprint:
            raise ValueError(
                "Refusing to resume Stage-1 build: input fingerprints, source "
                "revisions, caps, or producer options changed"
            )
    else:
        metadata = {
            "version": options.producer_version,
            "producer": options.producer,
            "producer_params": fingerprint_payload["producer_params"],
            "input_hashes": fingerprint_payload["input_hashes"],
            "source_revisions": {
                lang: spec.revision for lang, spec in sorted(sources.items())
            },
            "build_fingerprint": fingerprint,
            "unit": options.unit,
            "languages": {},
        }
        _atomic_json(metadata_path, metadata)

    writer = writer or ParquetShardWriter(options.compression)
    for lang in sorted(normalized_caps):
        source = sources[lang]
        cap = normalized_caps[lang]
        train_path = train_dir / f"{lang}.parquet"
        valid_path = valid_dir / f"{lang}.parquet"
        previous = metadata["languages"].get(lang, {})
        if _completed_artifacts_match(previous, train_path, valid_path):
            continue

        rng = _language_rng(options.seed, lang)
        rows: list[dict[str, Any]] = []
        chars = 0
        documents_seen = 0
        documents_contaminated = 0
        documents_empty = 0
        exhausted = True
        for text in document_provider(source):
            if not isinstance(text, str) or not text.strip():
                continue
            documents_seen += 1
            if contamination_matcher is not None and contamination_matcher(
                text, source.contamination_key
            ):
                documents_contaminated += 1
                continue
            if options.unit == "paragraph":
                emitted = paragraph_rows(text, lang)
            else:
                row = document_row(
                    text,
                    lang,
                    ratio=options.non_punctuation_sample_ratio,
                    rng=rng,
                    language_uses_punctuation=lang not in options.no_punctuation_languages,
                )
                emitted = [] if row is None else [row]
            if not emitted:
                documents_empty += 1
                continue
            for row in emitted:
                rows.append(row)
                chars += len(row["text"])
                if chars >= cap:
                    exhausted = False
                    break
            if chars >= cap:
                break
        if not rows:
            raise ValueError(f"Stage-1 source for {lang!r} produced an empty dataset")
        train_rows, valid_rows = deterministic_partition(
            rows,
            lang=lang,
            valid_ratio=options.valid_ratio,
            seed=options.seed,
        )
        if not train_rows or not valid_rows:
            raise ValueError(
                f"Stage-1 source for {lang!r} must produce at least two units "
                "so train and valid are both non-empty"
            )

        train_tmp = train_path.with_name(f".{train_path.name}.partial")
        valid_tmp = valid_path.with_name(f".{valid_path.name}.partial")
        for temporary in (train_tmp, valid_tmp):
            temporary.unlink(missing_ok=True)
        try:
            train_info = writer.write(train_tmp, train_rows)
            valid_info = writer.write(valid_tmp, valid_rows)
            # Completion is committed only after both durable artifacts exist.
            os.replace(train_tmp, train_path)
            os.replace(valid_tmp, valid_path)
        finally:
            train_tmp.unlink(missing_ok=True)
            valid_tmp.unlink(missing_ok=True)

        train_hashes = {
            hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            for row in train_rows
        }
        valid_hashes = {
            hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            for row in valid_rows
        }
        validation = {
            "required_columns": list(REQUIRED_COLUMNS),
            "no_empty_datasets": bool(train_rows and valid_rows),
            "split_overlap_count": len(train_hashes & valid_hashes),
            "cap_fill_ratio": chars / cap,
            "contamination": {
                "documents_seen": documents_seen,
                "documents_removed": documents_contaminated,
                "removal_ratio": (
                    documents_contaminated / documents_seen if documents_seen else 0.0
                ),
                "filter_enabled": contamination_matcher is not None,
            },
        }
        if validation["split_overlap_count"]:
            raise ValueError(f"Train/valid text overlap detected for {lang!r}")
        metadata["languages"][lang] = {
            "status": "complete",
            "source": _canonical(asdict(source)),
            "chars_budget": cap,
            "chars": chars,
            "chars_fill_ratio": chars / cap,
            "exhausted_source_before_budget": exhausted,
            "counts": {
                "documents_seen": documents_seen,
                "documents_contaminated": documents_contaminated,
                "documents_empty_after_transform": documents_empty,
                "units": len(rows),
                "train": len(train_rows),
                "valid": len(valid_rows),
            },
            "artifacts": {
                "train": {
                    "path": str(train_path),
                    **asdict(train_info),
                },
                "valid": {
                    "path": str(valid_path),
                    **asdict(valid_info),
                },
            },
            "validation": validation,
        }
        metadata["validation"] = {
            "languages_complete": sum(
                entry.get("status") == "complete"
                for entry in metadata["languages"].values()
            ),
            "languages_expected": len(normalized_caps),
        }
        _atomic_json(metadata_path, metadata)
    return metadata


def build_paragraphs(**kwargs: Any) -> dict[str, Any]:
    """Build paragraph-unit Stage-1 artifacts."""

    options: BuildOptions = kwargs["options"]
    kwargs["options"] = replace(options, unit="paragraph")
    return build_stage1_web(**kwargs)


def build_documents(**kwargs: Any) -> dict[str, Any]:
    """Build document-unit Stage-1 artifacts."""

    options: BuildOptions = kwargs["options"]
    kwargs["options"] = replace(options, unit="document")
    return build_stage1_web(**kwargs)
