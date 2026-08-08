#!/usr/bin/env python
"""Stream bounded FineWeb2 samples selected from the Stage-1 coverage plan.

The command defaults to a dry run. Execution writes one atomic JSONL file per
language/script and a hashed receipt. It never snapshots the complete dataset.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_rows(
    rows: list[dict],
    language_scripts: set[str],
    priorities: set[str],
) -> list[dict]:
    output = []
    for row in rows:
        if not row["hf_streaming_supported"]:
            continue
        if language_scripts and row["language_script"] not in language_scripts:
            continue
        if priorities and row["stage1_sampling_priority"] not in priorities:
            continue
        output.append(row)
    return output


def output_path(output: Path, language_script: str) -> Path:
    if not language_script.replace("_", "").isalnum():
        raise ValueError(f"Unsafe language/script identifier: {language_script}")
    return output / f"{language_script}.jsonl"


def sample_one(
    row: dict,
    target: Path,
    max_documents: int,
    remaining_bytes: int,
    seed: int,
    shuffle_buffer: int,
    revision: str,
) -> dict:
    from datasets import load_dataset

    if target.exists() and target.stat().st_size > 0:
        digest = hashlib.sha256()
        documents = 0
        with target.open("rb") as handle:
            while line := handle.readline():
                digest.update(line)
                documents += 1
        return {
            "language_script": row["language_script"],
            "path": str(target),
            "documents": documents,
            "bytes": target.stat().st_size,
            "sha256": digest.hexdigest(),
            "status": "cached",
        }
    stream = load_dataset(
        row["hf_dataset"],
        name=row["hf_config"],
        split=row["hf_split"],
        streaming=True,
        revision=revision,
    )
    if shuffle_buffer > 1:
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    written = 0
    documents = 0
    try:
        with temporary.open("wb") as handle:
            for example in stream:
                text = example.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                record = {
                    "text": text,
                    "lang": row["language_script"],
                    "source": "HuggingFaceFW/fineweb-2",
                    "source_id": example.get("id"),
                    "source_url": example.get("url"),
                    "language_score": example.get("language_score"),
                }
                payload = (
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
                if written + len(payload) > remaining_bytes:
                    raise RuntimeError("FineWeb2 sample exceeded cumulative byte budget")
                handle.write(payload)
                digest.update(payload)
                written += len(payload)
                documents += 1
                if documents >= max_documents:
                    break
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "language_script": row["language_script"],
        "path": str(target),
        "documents": documents,
        "bytes": written,
        "sha256": digest.hexdigest(),
        "status": "downloaded",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("data/manifests/fineweb2_stage1_sampling_plan_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/fineweb2-stage1-samples"),
    )
    parser.add_argument("--language-script", action="append", default=[])
    parser.add_argument("--priority", action="append", default=[])
    parser.add_argument("--max-documents-per-language", type=int, default=1_000)
    parser.add_argument("--max-download-mb", type=float, default=250.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--allow-many", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.max_documents_per_language <= 0:
        parser.error("--max-documents-per-language must be positive")
    if args.max_download_mb <= 0:
        parser.error("--max-download-mb must be positive")

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    rows = select_rows(
        plan["rows"],
        set(args.language_script),
        set(args.priority),
    )
    revisions = plan.get("dataset_revisions") or {}
    missing_revisions = sorted(
        {row["hf_dataset"] for row in rows if not revisions.get(row["hf_dataset"])}
    )
    if missing_revisions:
        parser.error(f"Sampling plan has no pinned revision for: {missing_revisions}")
    if args.execute and len(rows) > 25 and not args.allow_many:
        parser.error(
            f"Refusing to execute {len(rows)} languages without --allow-many"
        )
    preview = {
        "mode": "execute" if args.execute else "dry_run",
        "selected_language_script_pairs": len(rows),
        "max_documents_per_language": args.max_documents_per_language,
        "maximum_requested_documents": (
            len(rows) * args.max_documents_per_language
        ),
        "budget_mb": args.max_download_mb,
        "output": str(args.output),
        "dataset_revisions": revisions,
    }
    print(json.dumps(preview, indent=2))
    for row in rows[:25]:
        print(
            f"{row['language_script']:16s} "
            f"{row['stage1_sampling_priority']:30s} "
            f"{row['resource_bucket']}"
        )
    if len(rows) > 25:
        print(f"... {len(rows) - 25} additional language/script pairs")
    if not args.execute:
        return 0

    budget = int(args.max_download_mb * 1024 * 1024)
    downloaded = 0
    receipt_rows = []
    failures = 0
    for row in rows:
        target = output_path(args.output, row["language_script"])
        try:
            result = sample_one(
                row,
                target,
                min(
                    args.max_documents_per_language,
                    row.get(
                        "target_train_documents",
                        row["recommended_pilot_cap_documents"],
                    ),
                ),
                budget - downloaded,
                args.seed,
                args.shuffle_buffer,
                revisions[row["hf_dataset"]],
            )
            if result["status"] == "downloaded":
                downloaded += result["bytes"]
        except Exception as error:
            failures += 1
            result = {
                "language_script": row["language_script"],
                "status": "failed",
                "error": str(error),
            }
        receipt_rows.append(result)
        print(
            f"{result['status']:10s} {row['language_script']} "
            f"{result.get('documents', 0)} docs"
        )
        if downloaded >= budget:
            break
    receipt = {
        "version": "fineweb2_stage1_sample_receipt_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan": str(args.plan),
        "plan_sha256": file_sha256(args.plan),
        "dataset_revisions": revisions,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "budget_bytes": budget,
        "new_bytes": downloaded,
        "failures": failures,
        "rows": receipt_rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / "download_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"receipt: {receipt_path}")
    print(f"new download: {downloaded / 1024 / 1024:.2f} MiB")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
