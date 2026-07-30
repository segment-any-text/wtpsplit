"""Data selection helpers shared by Stage 2 and Stage 3 sentence training."""

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

LANG_SCRIPT_PATTERN = re.compile(r"^[a-z]{2,3}_[A-Z][a-z]{3}$")
DEFAULT_TRAINING_DATASET_PRIORITY = (
    "ud",
    "tatoeba",
    "opus100",
    "nllb",
    "projected",
)
SUPPORTED_TRAINING_DATASETS = frozenset(DEFAULT_TRAINING_DATASET_PRIORITY)
CORRUPTION_VARIANTS = ("corrupted-asr", "corrupted-social-media")


def is_monolingual_language_code(language_code: str) -> bool:
    """Accept plain language IDs and ISO language/script IDs, but not pairs.

    Stage 3 historically excludes identifiers such as ``en-de`` and ``en_de``
    because they denote code-switching data. Stage 2 sources use identifiers
    such as ``deu_Latn`` and ``bod_Tibt``; their four-letter ISO 15924 suffix
    is a script, not a second language.
    """

    if "-" in language_code:
        return False
    if "_" not in language_code:
        return True
    return LANG_SCRIPT_PATTERN.fullmatch(language_code) is not None


def select_training_dataset(
    sentence_datasets: Mapping[str, Any],
    requested_dataset: str | None = None,
) -> str | None:
    """Select a dataset that has a non-empty flat ``meta.train_data`` list."""

    if requested_dataset is not None and requested_dataset not in SUPPORTED_TRAINING_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_TRAINING_DATASETS))
        raise ValueError(f"Unsupported training dataset {requested_dataset!r}; choose one of: {supported}.")

    candidates = (requested_dataset,) if requested_dataset is not None else DEFAULT_TRAINING_DATASET_PRIORITY
    for dataset_name in candidates:
        dataset = sentence_datasets.get(dataset_name)
        if not isinstance(dataset, Mapping):
            continue
        meta = dataset.get("meta")
        if not isinstance(meta, Mapping):
            continue
        train_data = meta.get("train_data")
        if isinstance(train_data, Sequence) and not isinstance(train_data, (str, bytes)) and train_data:
            return dataset_name
    return None


def _bounded(sentences: Sequence[str], limit: int | None) -> list[str]:
    if limit is None:
        return list(sentences)
    return list(sentences[:limit])


def _training_data(
    sentence_datasets: Mapping[str, Any],
    dataset_name: str,
    *,
    language_code: str,
    limit: int | None,
) -> list[str]:
    try:
        train_data = sentence_datasets[dataset_name]["meta"]["train_data"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Language {language_code!r} is missing meta.train_data for dataset {dataset_name!r}."
        ) from error
    if not isinstance(train_data, Sequence) or isinstance(train_data, (str, bytes)) or not train_data:
        raise ValueError(
            f"Language {language_code!r} has no training sentences for dataset {dataset_name!r}."
        )
    if not all(isinstance(sentence, str) for sentence in train_data):
        raise ValueError(
            f"Language {language_code!r} dataset {dataset_name!r} must contain strings."
        )
    # Keep legacy Stage 3 extraction unchanged so new experiments remain
    # comparable with completed pilots. New projected corpora have a stricter
    # contract because an empty projected sentence carries no weak label.
    if dataset_name == "projected" or dataset_name.startswith("projected-"):
        nonempty_train_data = [sentence for sentence in train_data if sentence]
        if len(nonempty_train_data) != len(train_data):
            raise ValueError(
                f"Language {language_code!r} dataset {dataset_name!r} contains empty sentences."
            )
    return _bounded(train_data, limit)


def prepare_sentence_datasets(
    all_data: Mapping[str, Any],
    *,
    selected_languages: set[str] | None,
    requested_training_dataset: str | None,
    no_sm_corruption: bool,
    max_train_sentences_per_dataset: int | None,
    max_eval_instances_per_dataset: int,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, list[Any]]]]:
    """Validate and extract the nested ``.pth`` corpus used by ``train_SM``."""

    train_sentences: defaultdict[str, defaultdict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    test_sentences: defaultdict[str, defaultdict[str, list[Any]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for language_code, language_data in all_data.items():
        if selected_languages is not None and language_code not in selected_languages:
            continue
        if not isinstance(language_data, Mapping) or not isinstance(language_data.get("sentence"), Mapping):
            continue
        sentence_datasets = language_data["sentence"]
        if is_monolingual_language_code(language_code):
            training_dataset = select_training_dataset(sentence_datasets, requested_training_dataset)
            if training_dataset is not None:
                clean = _training_data(
                    sentence_datasets,
                    training_dataset,
                    language_code=language_code,
                    limit=max_train_sentences_per_dataset,
                )
                if (
                    training_dataset == "ud"
                    and max_train_sentences_per_dataset is None
                    and len(clean) < 10_000
                ):
                    clean *= 10_000 // len(clean) + 1
                train_sentences[language_code]["uncorrupted"].extend(clean)

                if not no_sm_corruption:
                    for corruption in CORRUPTION_VARIANTS:
                        corrupted_dataset = f"{training_dataset}-{corruption}"
                        if corrupted_dataset not in sentence_datasets:
                            raise ValueError(
                                f"Language {language_code!r} dataset {training_dataset!r} is missing "
                                f"{corrupted_dataset!r}; provide the corruption variant or set "
                                "`no_sm_corruption=true`."
                            )
                        corrupted = _training_data(
                            sentence_datasets,
                            corrupted_dataset,
                            language_code=language_code,
                            limit=max_train_sentences_per_dataset,
                        )
                        if (
                            training_dataset == "ud"
                            and max_train_sentences_per_dataset is None
                            and len(corrupted) < 5_000
                        ):
                            corrupted *= 10_000 // len(corrupted) + 1
                        train_sentences[language_code][corruption].extend(corrupted)

        for dataset_name, dataset in sentence_datasets.items():
            if dataset_name.startswith(("short-sequences", "legal")):
                continue
            if not isinstance(dataset, Mapping):
                continue
            test_data = dataset.get("data")
            if not isinstance(test_data, Sequence) or isinstance(test_data, (str, bytes)):
                continue
            test_sentences[language_code][dataset_name].extend(
                test_data[:max_eval_instances_per_dataset]
            )

    if selected_languages is not None:
        missing_languages = selected_languages - set(train_sentences)
        if missing_languages:
            raise ValueError(f"Requested languages produced no training data: {sorted(missing_languages)}")
    if not train_sentences:
        dataset_note = (
            f" for requested dataset {requested_training_dataset!r}"
            if requested_training_dataset is not None
            else ""
        )
        raise ValueError(f"Corpus produced no monolingual training data{dataset_note}.")

    return (
        {
            language_code: {
                dataset_name: list(sentences) for dataset_name, sentences in datasets.items()
            }
            for language_code, datasets in train_sentences.items()
        },
        {
            language_code: {
                dataset_name: list(instances) for dataset_name, instances in datasets.items()
            }
            for language_code, datasets in test_sentences.items()
        },
    )
