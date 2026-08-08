#!/usr/bin/env python
"""Download selected UD 2.18 files from an acquisition manifest.

Dry run unless ``--execute`` is set. Writes through atomic ``.part`` files, stops
at a cumulative byte budget, and records SHA-256 hashes in a receipt.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.error
import urllib.request

CHUNK_SIZE = 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def select_rows(
    rows: list[dict],
    selection: str,
    repositories: set[str],
    language_scripts: set[str],
) -> list[dict]:
    selected = []
    for row in rows:
        if repositories and row["repository"] not in repositories:
            continue
        if language_scripts and not (
            language_scripts & set(row["language_script_candidates"])
        ):
            continue
        if selection == "primary" and not row["provisional_primary_for_language"]:
            continue
        if selection == "p0-p1" and not row["acquisition_priority"].startswith(
            ("P0", "P1")
        ):
            continue
        selected.append(row)
    return selected


def safe_target(output: Path, repository: str, filename: str) -> Path:
    if not re.fullmatch(r"UD_[A-Za-z0-9_.-]+", repository):
        raise ValueError(f"Unsafe repository name: {repository}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", filename):
        raise ValueError(f"Unsafe filename: {filename}")
    return output / repository / filename


def stream_download(
    url: str,
    target: Path,
    remaining_bytes: int,
) -> tuple[int, str, str]:
    if target.exists() and target.stat().st_size > 0:
        digest = file_sha256(target)
        return target.stat().st_size, digest, "cached"
    request = urllib.request.Request(url, headers={"User-Agent": "mmsat-ud-downloader"})
    with urllib.request.urlopen(request, timeout=90) as response:
        declared = int(response.headers.get("Content-Length", "0"))
        if declared and declared > remaining_bytes:
            raise RuntimeError(
                f"Download would exceed remaining budget: {declared} > {remaining_bytes}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        digest = hashlib.sha256()
        written = 0
        try:
            with temporary.open("wb") as handle:
                while chunk := response.read(CHUNK_SIZE):
                    written += len(chunk)
                    if written > remaining_bytes:
                        raise RuntimeError("Download exceeded cumulative byte budget")
                    handle.write(chunk)
                    digest.update(chunk)
            temporary.replace(target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return written, digest.hexdigest(), "downloaded"


def candidate_files(row: dict, splits: list[str], metadata: bool) -> list[tuple[str, list[str]]]:
    stem = row["treebank_stem"]
    files: list[tuple[str, list[str]]] = []
    for split in splits:
        if split == "evaluation":
            files.append(
                (
                    "evaluation",
                    [row["evaluation_test_url"], row["evaluation_dev_fallback_url"]],
                )
            )
        else:
            files.append(
                (
                    split,
                    [
                        (
                            f"https://raw.githubusercontent.com/UniversalDependencies/"
                            f"{row['repository']}/{row['release_ref']}/"
                            f"{stem}-ud-{split}.conllu"
                        )
                    ],
                )
            )
    if metadata:
        base = (
            f"https://raw.githubusercontent.com/UniversalDependencies/"
            f"{row['repository']}/{row['release_ref']}"
        )
        files.extend(
            [
                (
                    "license",
                    list(
                        dict.fromkeys(
                            [
                                row.get("license_url_verified"),
                                row["license_url"],
                                f"{base}/LICENSE.md",
                                f"{base}/LICENSE",
                            ]
                        )
                    ),
                ),
                (
                    "readme",
                    list(
                        dict.fromkeys(
                            [
                                row.get("readme_url_verified"),
                                row["readme_url"],
                                f"{base}/README.txt",
                            ]
                        )
                    ),
                ),
            ]
        )
    return [
        (kind, [url for url in urls if url])
        for kind, urls in files
    ]


def filename_from_url(url: str) -> str:
    return url.rsplit("/", 1)[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("data/manifests/ud_2_18_acquisition_plan_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/ud-treebanks-v2.18-selective"),
    )
    parser.add_argument(
        "--selection",
        choices=["primary", "p0-p1", "all"],
        default="primary",
    )
    parser.add_argument("--repository", action="append", default=[])
    parser.add_argument("--language-script", action="append", default=[])
    parser.add_argument(
        "--split",
        action="append",
        choices=["evaluation", "test", "dev", "train"],
        dest="splits",
    )
    parser.add_argument("--metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-download-mb", type=float, default=250.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.max_download_mb <= 0:
        parser.error("--max-download-mb must be positive")

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    rows = select_rows(
        plan["rows"],
        args.selection,
        set(args.repository),
        set(args.language_script),
    )
    splits = args.splits or ["evaluation"]
    requests = [
        {
            "repository": row["repository"],
            "language_script_candidates": row["language_script_candidates"],
            "kind": kind,
            "candidate_urls": urls,
        }
        for row in rows
        for kind, urls in candidate_files(row, splits, args.metadata)
    ]
    preview = {
        "mode": "execute" if args.execute else "dry_run",
        "release": plan["release"],
        "selected_treebanks": len(rows),
        "file_requests": len(requests),
        "budget_mb": args.max_download_mb,
        "output": str(args.output),
    }
    print(json.dumps(preview, indent=2))
    if not args.execute:
        for request in requests[:20]:
            print(
                f"{request['repository']:40s} {request['kind']:10s} "
                f"{request['candidate_urls'][0]}"
            )
        if len(requests) > 20:
            print(f"... {len(requests) - 20} additional file requests")
        return 0

    budget = int(args.max_download_mb * 1024 * 1024)
    downloaded = 0
    receipt_rows = []
    failures = 0
    for request in requests:
        result = None
        errors = []
        for url in request["candidate_urls"]:
            target = safe_target(
                args.output,
                request["repository"],
                filename_from_url(url),
            )
            try:
                size, digest, status = stream_download(
                    url,
                    target,
                    budget - downloaded,
                )
            except urllib.error.HTTPError as error:
                errors.append(f"{url}: HTTP {error.code}")
                if error.code == 404:
                    continue
                break
            except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
                errors.append(f"{url}: {error}")
                break
            if status == "downloaded":
                downloaded += size
            result = {
                **request,
                "url": url,
                "path": str(target),
                "bytes": size,
                "sha256": digest,
                "status": status,
            }
            break
        if result is None:
            failures += 1
            result = {**request, "status": "failed", "errors": errors}
        receipt_rows.append(result)
        print(f"{result['status']:10s} {request['repository']} {request['kind']}")

    receipt = {
        "version": "ud_selective_download_receipt_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan": str(args.plan),
        "plan_sha256": file_sha256(args.plan),
        "release": plan["release"],
        "selection": args.selection,
        "splits": splits,
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
