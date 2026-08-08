from pathlib import Path

from scripts.build_stage2_ud_evaluation import build_packet, packet_rows


def test_packet_rows_use_exact_dev_contract():
    rows = packet_rows(
        ["One.", "Two.", "Three."],
        language_script="eng_Latn",
        repository="UD_English-EWT",
        release="r2.18",
        license_name="CC-BY-SA-4.0",
        split="dev",
        sentences_per_document=3,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["reference_type"] == "exact"
    assert row["split"] == "dev"
    assert row["text"] == "One. Two. Three."
    assert row["gold_offsets"] == [5, 10]


def test_build_packet_reads_local_pinned_treebank(tmp_path: Path):
    repo = tmp_path / "UD_Test-TB"
    repo.mkdir()
    payload = "# text = First.\n\n# text = Second.\n\n"
    (repo / "xx_tb-ud-dev.conllu").write_text(payload, encoding="utf-8")
    (repo / "xx_tb-ud-test.conllu").write_text(payload, encoding="utf-8")
    selection = {
        "rows": [
            {
                "selection_status": "frozen",
                "language_script": "xxx_Latn",
                "repository": "UD_Test-TB",
                "treebank_stem": "xx_tb",
                "release_ref": "r2.18",
                "source_license": "CC-BY-4.0",
            }
        ]
    }
    rows, sources = build_packet(selection, tmp_path, 2)
    assert [row["split"] for row in rows] == ["dev", "test"]
    assert sources[0]["splits"]["dev"]["sentences"] == 2
    assert len(sources[0]["splits"]["dev"]["sha256"]) == 64
