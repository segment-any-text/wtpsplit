from collections import defaultdict

from wtpsplit.utils import Constants


def tokenize_and_get_labels(sentences, tokenizer, separator):
    joined_sentence = separator.join(sentences)
    sentence_lengths = [len(sentence) for sentence in sentences]

    # calculate where each sentence ends
    sentence_end_positions = [sum(sentence_lengths[: i + 1]) + i * len(separator) - 1 for i in range(len(sentences))]

    # tokenize whole text at once
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
    if not tokens:
        return [], []

    # labels
    sentence_ending_labels = [0] * len(tokens)

    # last token of each sentence to 1
    sentence_index = 0
    for i, (start, end) in enumerate(offsets):
        if sentence_index < len(sentence_end_positions) and end > sentence_end_positions[sentence_index]:
            sentence_ending_labels[i - 1] = 1
            sentence_index += 1
        if sentence_index >= len(sentence_end_positions):
            break

    # Make sure the last token of the last sentence is marked if it wasn't already
    sentence_ending_labels[-1] = 1

    return tokenized_input["input_ids"], sentence_ending_labels


def pack_sentences(examples, block_size, tokenizer, overflow_size=0):
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

        # batch tokenize sentences
        tokenized_sentences = tokenizer(sentences, add_special_tokens=False, verbose=False, padding=False)
        input_ids_list = tokenized_sentences["input_ids"]

        for sentence, input_ids in zip(sentences, input_ids_list):
            if not sentence or sentence.isnumeric():
                continue
            num_sentence_tokens = len(input_ids)

            # Allow exceeding block size slightly to avoid underfilling blocks
            if token_count + num_sentence_tokens > block_size + overflow_size:
                # limit exceeded, process the current block
                if one_block_sentences:
                    input_ids, labels = tokenize_and_get_labels(one_block_sentences, tokenizer, separator)
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
            input_ids, labels = tokenize_and_get_labels(one_block_sentences, tokenizer, separator)
            all_input_blocks.append(input_ids)
            all_label_blocks.append(labels)
            all_langs.append(current_lang)

    # only return label indices, ie == 1 --> save memory
    all_label_blocks = [[i for i, label in enumerate(labels) if label == 1] for labels in all_label_blocks]

    return {
        "input_ids": all_input_blocks,
        "labels": all_label_blocks,
        "lang": all_langs,
    }
