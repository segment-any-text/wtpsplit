"""Evaluate few-shot SaT adaptation on held-out BOUQuET sentences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

from wtpsplit import SaT
from wtpsplit.evaluation import evaluate_sentences, preprocess_sentence
from wtpsplit.utils import Constants


def load_bouquet_sentences(language: str, limit: int) -> list[str]:
    dataset = load_dataset("facebook/bouquet", "sentence_level", split="test", streaming=True)
    sentences = []
    seen = set()
    for row in dataset:
        if row["src_lang"] != language:
            continue
        sentence = preprocess_sentence(row["src_text"])
        if sentence and "\n" not in sentence and sentence not in seen:
            seen.add(sentence)
            sentences.append(sentence)
        if len(sentences) >= limit:
            break
    if len(sentences) < limit:
        raise RuntimeError(f"BOUQuET provided only {len(sentences)} usable {language} sentences; need {limit}.")
    return sentences


def evaluate_model(model: SaT, sentences: list[str], language: str, document_size: int) -> dict[str, float]:
    scores = []
    recalls = []
    precisions = []
    for start in range(0, len(sentences), document_size):
        gold = sentences[start : start + document_size]
        if len(gold) < 2:
            continue
        text = Constants.SEPARATORS[language].join(gold)
        predicted = model.split(text)
        score, info = evaluate_sentences(language, gold, predicted)
        scores.append(score)
        recalls.append(info["recall"])
        precisions.append(info["precision"])
    return {
        "f1": float(np.mean(scores)),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "documents": len(scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sat-3l-sm")
    parser.add_argument("--language", default="de")
    parser.add_argument("--bouquet-language", default="deu_Latn")
    parser.add_argument("--shots", default="10,50,100")
    parser.add_argument("--eval-sentences", type=int, default=200)
    parser.add_argument("--document-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    shots = sorted({int(value) for value in args.shots.split(",")})
    if not shots or shots[0] < 2:
        raise ValueError("`--shots` must contain integers of at least 2.")

    sentences = load_bouquet_sentences(args.bouquet_language, max(shots) + args.eval_sentences)
    held_out = sentences[max(shots) :]

    base = SaT(args.model, device=args.device)
    results = {
        "model": args.model,
        "language": args.language,
        "bouquet_language": args.bouquet_language,
        "eval_sentences": len(held_out),
        "epochs": args.epochs,
        "base": evaluate_model(base, held_out, args.language, args.document_size),
        "adapted": {},
    }

    for count in shots:
        adapted = SaT(args.model, device=args.device)
        adapted.adapt(
            sentences[:count],
            language=args.language,
            epochs=args.epochs,
            show_progress=False,
        )
        metrics = evaluate_model(adapted, held_out, args.language, args.document_size)
        metrics["delta_f1"] = metrics["f1"] - results["base"]["f1"]
        metrics["first_loss"] = adapted.adaptation_history[0]
        metrics["final_loss"] = adapted.adaptation_history[-1]
        results["adapted"][str(count)] = metrics

    rendered = json.dumps(results, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
