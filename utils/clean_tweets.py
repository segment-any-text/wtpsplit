import re
import torch


def remove_emojis_and_special_chars(text):
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f700-\U0001f77f"  # alchemical symbols
        "\U0001f780-\U0001f7ff"  # Geometric Shapes Extended
        "\U0001f800-\U0001f8ff"  # Supplemental Arrows-C
        "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
        "\U0001fa00-\U0001fa6f"  # Chess Symbols
        "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027b0"  # Dingbats
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)
    text = re.sub(r"[:;=Xx][\-oO\']*[\)\(\[\]DdPp3><\|\\\/]", "", text)
    return text


def transform_data(data):
    def pair_sentences(sequences):
        paired_sequences = []
        for sequence in sequences:
            processed_sequence = []
            for sentence in sequence:
                words = sentence.strip().split()
                filtered_words = [
                    remove_emojis_and_special_chars(word)
                    for word in words
                    if not (word.startswith("http") or word.startswith("#") or word.startswith("@"))
                ]
                cleaned_sentence = " ".join(filtered_words)  # fine for our langs.
                if cleaned_sentence and len(cleaned_sentence.split()) > 0:
                    processed_sequence.append(cleaned_sentence.strip())
            if processed_sequence and len(processed_sequence) < 6:
                paired_sequences.append(processed_sequence)
        return paired_sequences

    transformed_data = {}
    for lang_code, lang_data in data.items():
        transformed_data[lang_code] = {}
        for content_type, datasets in lang_data.items():
            if content_type != "sentence":
                continue
            transformed_data[lang_code] = {}
            transformed_data[lang_code][content_type] = {}
            for dataset_name, content in datasets.items():
                if "short" not in dataset_name:
                    continue
                transformed_data[lang_code][content_type][dataset_name] = {
                    "meta": {"train_data": pair_sentences(content["meta"]["train_data"])},
                    "data": pair_sentences(content["data"]),
                }

    return transformed_data


data = torch.load("data/all_data.pth")

transformed_data = transform_data(data)
torch.save(transformed_data, "data/all_data_tweets_cleaned.pth")
