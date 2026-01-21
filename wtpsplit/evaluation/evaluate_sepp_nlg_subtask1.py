import json
import os
import sys

from sklearn.metrics import classification_report

from wtpsplit.utils import Constants


def evaluate_subtask1(splits, langs, prediction_dir: str, supervisions, include_n_documents) -> None:
    """
    Mirrors the original SEPP-NLG 2021 Shared Task evaluation function
    https://sites.google.com/view/sentence-segmentation
    """

    results = {}
    avg_holder = {}
    for supervision in supervisions:
        avg_holder[supervision] = 0
    for lang_code in langs:
        results[lang_code] = {}
        for split in splits:
            results[lang_code][split] = {}
            for supervision in supervisions:
                print(lang_code, split, supervision)
                relevant_dir = Constants.ROOT_DIR.parent / "data/sepp_nlg_2021_data" / lang_code / split

                all_gt_labels, all_predicted_labels = [], []
                fnames = sorted([f for f in relevant_dir.glob("*.tsv") if f.is_file()])[:include_n_documents]
                gt_tsv_files = [
                    fname
                    for fname in fnames
                    if str(fname).startswith(str(relevant_dir)) and str(fname).endswith(".tsv")
                ]

                for _, gt_tsv_file in enumerate(gt_tsv_files, 0):
                    basename = os.path.basename(gt_tsv_file)

                    with open(gt_tsv_file, encoding="utf-8") as f:
                        lines = f.read().strip().split("\n")
                        rows = [line.split("\t") for line in lines]
                        gt_labels = [row[1] for row in rows]

                    with open(
                        os.path.join(
                            Constants.CACHE_DIR, "ted2020", prediction_dir, lang_code, split, supervision, basename
                        ),
                        "r",
                        encoding="utf8",
                    ) as f:
                        lines = f.read().strip().split("\n")
                        rows = [line.split("\t") for line in lines]
                        pred_labels = [row[1] for row in rows]

                    assert len(gt_labels) == len(pred_labels), (
                        f"unequal no. of labels for files {gt_tsv_file} and {os.path.join(prediction_dir, lang_code, split, basename)}"
                    )
                    all_gt_labels.extend(gt_labels)
                    all_predicted_labels.extend(pred_labels)

                eval_result = classification_report(all_gt_labels, all_predicted_labels, output_dict=True)
                print(eval_result["1"]["f1-score"])
                avg_holder[supervision] += eval_result["1"]["f1-score"]
                results[lang_code][split][supervision] = eval_result
    results["avg"] = {}
    for supervision in supervisions:
        avg_holder[supervision] /= len(langs)
        results["avg"][supervision] = avg_holder[supervision]
    print(avg_holder)
    json.dump(
        results,
        open(
            Constants.CACHE_DIR / "ted2020" / f"{prediction_dir}_TED.json",
            "w",
        ),
        indent=4,
    )
    print(Constants.CACHE_DIR / "ted2020" / f"{prediction_dir}_TED.json")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate subtask 1 of SEPP-NLG 2021")
    parser.add_argument(
        "--include_languages",
        help="target language ('en', 'de', 'fr', 'it'; i.e. one of the subfolders in the zip file's main folder)",
        default=["fr", "de", "en", "it"],
        nargs="+",
    )
    parser.add_argument(
        "--splits",
        help="split to be evaluated (usually 'dev', 'test'), subfolder of 'lang'",
        default=["test", "surprise_test"],
        nargs="+",
    )
    parser.add_argument(
        "--prediction_dir",
        help="path to folder containing the prediction TSVs (language and test set folder names are appended automatically)",
    )
    parser.add_argument(
        "--supervision",
        help="u, t, punct",
        default=["u", "t", "punct"],
        nargs="+",
    )
    parser.add_argument("--include_n_documents", default=sys.maxsize)
    args = parser.parse_args()
    results = evaluate_subtask1(args.splits, args.include_languages, args.prediction_dir, args.supervision)
