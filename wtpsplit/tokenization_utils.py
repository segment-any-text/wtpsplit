import numpy as np
from wtpsplit.utils import Constants


def tokenize_and_get_labels(sentences, tokenizer, separator, lang_code):
    joined_sentence = ""
    sentence_start_positions = []
    current_position = 0

    for sentence in sentences:
        if joined_sentence:
            joined_sentence += separator
            current_position += len(separator)
        start_position = current_position
        joined_sentence += sentence
        current_position += len(sentence)
        sentence_start_positions.append(start_position + len(sentence) - 1)

    tokenized_input = tokenizer(
        joined_sentence,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
        padding=False,
    )

    tokens = tokenized_input.tokens()
    offsets = tokenized_input["offset_mapping"]
    sentence_ending_labels = [0] * len(tokens)

    sentence_index = 0
    for i, (token_start, token_end) in enumerate(offsets):
        if token_start > sentence_start_positions[sentence_index]:
            sentence_ending_labels[i - 1] = 1
            sentence_index += 1
            # if any(start < token_end for start in sentence_start_positions if start >= token_start):
            #     print(tokens[i - 2 : i + 3])

    # assert sum(sentence_ending_labels) == len(sentence_start_positions)

    return tokenized_input["input_ids"], sentence_ending_labels


def pack_sentences(examples, block_size, tokenizer, underflow_size=0, min_sentence_length=10):
    all_input_blocks = []
    all_label_blocks = []
    all_langs = []

    # group by langs first
    lang_grouped_examples = {lang: [] for lang in set(examples["lang"])}
    for sentence, lang in zip(examples["text"], examples["lang"]):
        lang_grouped_examples[lang].append(sentence.strip("\n"))

    for current_lang, sentences in lang_grouped_examples.items():
        separator = Constants.SEPARATORS.get(current_lang, " ")
        token_count, one_block_sentences = 0, []

        # tokenization mapping gets problematic in such instances
        sentences = [sentence.replace("\ufffd", "").strip() for sentence in sentences]
        sentences = [sentence for sentence in sentences if len(sentence) > min_sentence_length]
        if not sentences:
            continue

        # batch tokenize sentences
        tokenized_sentences = tokenizer(sentences, add_special_tokens=False, verbose=False, padding=False)
        input_ids_list = tokenized_sentences["input_ids"]

        for sentence, input_ids in zip(sentences, input_ids_list):
            if not sentence or sentence.isnumeric():
                continue
            num_sentence_tokens = len(input_ids)

            # check if block limit is exceeded
            if token_count > block_size - underflow_size:
                # limit exceeded, process the current block
                if one_block_sentences:
                    input_ids, labels = tokenize_and_get_labels(one_block_sentences, tokenizer, separator, current_lang)
                    all_input_blocks.append(input_ids)
                    all_label_blocks.append(labels)
                    all_langs.append(current_lang)
                # reset
                token_count, one_block_sentences = 0, []

            # add sentence to block
            one_block_sentences.append(sentence)
            token_count += num_sentence_tokens

        # ensure last batch of sentences is processed
        if one_block_sentences:
            input_ids, labels = tokenize_and_get_labels(one_block_sentences, tokenizer, separator, current_lang)
            all_input_blocks.append(input_ids)
            all_label_blocks.append(labels)
            all_langs.append(current_lang)

    # only return label indices, ie == 1 --> save memory
    all_label_blocks = [[i for i, label in enumerate(labels) if label == 1] for labels in all_label_blocks]

    # TODO: in addition, truncate blocks here already? (storage reasons)
    return {
        "input_ids": all_input_blocks,
        "labels": all_label_blocks,
        "lang": all_langs,
    }
