import json
from pathlib import Path
import random

import pytest

from wtpsplit.data_acquisition.stage1_web import (
    BuildOptions,
    JsonlShardWriter,
    SourceSpec,
    apply_punctuation_mixture,
    build_documents,
    build_paragraphs,
    deterministic_partition,
    document_row,
    ends_with_punctuation,
    paragraph_rows,
    resolve_web_sources,
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def fixture_source(lang: str = "xx") -> SourceSpec:
    return SourceSpec(
        lang=lang,
        corpus="fineweb2",
        dataset="fixture/web",
        config=f"{lang}_Latn",
        revision="fixture-revision-1",
        language_script=f"{lang}_Latn",
        contamination_key=f"{lang}_Latn",
    )


def test_source_resolution_applies_script_remap_before_plan_lookup():
    mapping = {
        "rows": [
            {"sat_lang": "ar", "fineweb2_language_script": "arb_Latn"},
            {
                "sat_lang": "en",
                "fineweb2_language_script": None,
                "fineweb_english": {
                    "hf_dataset": "HuggingFaceFW/fineweb",
                    "hf_config": "sample-10BT",
                    "local_script_id": "eng_FineWeb",
                },
            },
        ]
    }
    remaps = {
        "remaps": [
            {
                "sat_lang": "ar",
                "from_fineweb2_language_script": "arb_Latn",
                "to_fineweb2_language_script": "arb_Arab",
            }
        ]
    }
    plan = {
        "rows": [
            {
                "language_script": "arb_Arab",
                "hf_dataset": "HuggingFaceFW/fineweb-2",
                "hf_config": "arb_Arab",
                "hf_split": "train",
            }
        ]
    }

    sources = resolve_web_sources(
        corpus="fineweb2",
        language_map=mapping,
        plan=plan,
        script_remaps=remaps,
        revisions={"HuggingFaceFW/fineweb-2": "rev-a"},
    )

    assert sources["ar"].config == "arb_Arab"
    assert sources["ar"].contamination_key == "arb_Arab"
    assert sources["ar"].revision == "rev-a"
    assert sources["en"].dataset == "HuggingFaceFW/fineweb"


def test_mc4_resolution_uses_current_hub_config_alias():
    sources = resolve_web_sources(
        corpus="mc4",
        language_map={"rows": [{"sat_lang": "he"}, {"sat_lang": "en"}]},
    )
    assert sources["he"].dataset == "allenai/c4"
    assert sources["he"].config == "iw"
    assert sources["en"].config == "en"


def test_partition_and_punctuation_flags_are_deterministic():
    rows = [
        {
            "text": f"row {index}{'.' if index % 2 else ''}",
            "lang": "xx",
            "ends_with_punctuation": bool(index % 2),
        }
        for index in range(20)
    ]
    first = deterministic_partition(rows, lang="xx", valid_ratio=0.2, seed=17)
    second = deterministic_partition(rows, lang="xx", valid_ratio=0.2, seed=17)
    assert first == second
    assert first[0] and first[1]
    assert ends_with_punctuation("Hello!\n")
    assert ends_with_punctuation("値。")
    assert not ends_with_punctuation("Hello\n")


def test_paragraph_and_document_transforms_remain_distinct():
    text = "First.\nNo ending\nLast!"
    paragraphs = paragraph_rows(text, "xx")
    assert [row["text"] for row in paragraphs] == [
        "First.\n",
        "No ending\n",
        "Last!\n",
    ]
    assert [row["ends_with_punctuation"] for row in paragraphs] == [
        True,
        False,
        True,
    ]

    document = document_row(
        text,
        "xx",
        ratio=0.0,
        rng=random.Random(3),
        language_uses_punctuation=True,
    )
    assert document == {
        "text": "First.\nLast!\n",
        "ends_with_punctuation": True,
        "lang": "xx",
    }
    assert apply_punctuation_mixture(
        ["Only no punctuation\n"],
        ratio=0.0,
        rng=random.Random(3),
        language_uses_punctuation=True,
    ) == ["Only no punctuation\n"]


@pytest.mark.parametrize(
    ("task", "unit", "expected_rows"),
    [
        (build_paragraphs, "paragraph", 6),
        (build_documents, "document", 3),
    ],
)
def test_build_tasks_write_per_language_artifacts_and_metadata(
    tmp_path: Path, task, unit: str, expected_rows: int
):
    source = fixture_source()
    documents = [
        "Alpha.\nBeta?",
        "Gamma!\nDelta.",
        "Epsilon?\nZeta!",
    ]

    metadata = task(
        caps={"xx": 10_000},
        sources={"xx": source},
        options=BuildOptions(
            output_dir=tmp_path,
            unit=unit,
            valid_ratio=0.25,
            non_punctuation_sample_ratio=None,
        ),
        document_provider=lambda _: iter(documents),
        writer=JsonlShardWriter(),
        input_hashes={"fixture": "sha256:fixture-v1"},
    )

    train_path = tmp_path / "train" / "xx.parquet"
    valid_path = tmp_path / "valid" / "xx.parquet"
    train = read_jsonl(train_path)
    valid = read_jsonl(valid_path)
    entry = metadata["languages"]["xx"]
    assert len(train) + len(valid) == expected_rows
    assert train and valid
    assert entry["status"] == "complete"
    assert entry["source"]["revision"] == "fixture-revision-1"
    assert entry["artifacts"]["train"]["sha256"]
    assert entry["counts"]["train"] == len(train)
    assert entry["validation"]["no_empty_datasets"]
    assert entry["validation"]["contamination"]["documents_seen"] == 3


def test_resume_skips_matching_completed_shards_and_rejects_stale_inputs(
    tmp_path: Path,
):
    source = fixture_source()
    documents = ["One.\nTwo!", "Three?\nFour.", "Five!\nSix?"]
    options = BuildOptions(
        output_dir=tmp_path,
        unit="paragraph",
        valid_ratio=0.2,
    )
    kwargs = {
        "caps": {"xx": 1_000},
        "sources": {"xx": source},
        "options": options,
        "writer": JsonlShardWriter(),
        "input_hashes": {"mapping": "map-v1"},
    }
    build_paragraphs(
        **kwargs,
        document_provider=lambda _: iter(documents),
    )

    def must_not_open(_: SourceSpec):
        raise AssertionError("completed source should not be reopened")

    resumed = build_paragraphs(**kwargs, document_provider=must_not_open)
    assert resumed["languages"]["xx"]["status"] == "complete"

    with pytest.raises(ValueError, match="Refusing to resume"):
        build_paragraphs(
            **{
                **kwargs,
                "input_hashes": {"mapping": "map-v2"},
                "document_provider": must_not_open,
            }
        )


def test_contamination_is_filtered_before_paragraph_expansion(tmp_path: Path):
    source = fixture_source()
    documents = ["CONTAMINATED.\nAlso contaminated!", "Clean.\nUsable!"]
    metadata = build_paragraphs(
        caps={"xx": 1_000},
        sources={"xx": source},
        options=BuildOptions(
            output_dir=tmp_path,
            unit="paragraph",
            valid_ratio=0.5,
        ),
        document_provider=lambda _: iter(documents),
        contamination_matcher=lambda text, _: text.startswith("CONTAMINATED"),
        writer=JsonlShardWriter(),
    )
    rows = read_jsonl(tmp_path / "train" / "xx.parquet") + read_jsonl(
        tmp_path / "valid" / "xx.parquet"
    )
    assert {row["text"] for row in rows} == {"Clean.\n", "Usable!\n"}
    contamination = metadata["languages"]["xx"]["validation"]["contamination"]
    assert contamination["documents_removed"] == 1
