import os
from transformers import AutoTokenizer
from tokenizers import AddedToken
from wtpsplit.utils import Constants, LabelArgs

def get_subword_label_dict(label_args, tokenizer):
    label_dict = {}

    n_unks = 0
    # Map auxiliary characters to token IDs with labels
    for i, c in enumerate(Constants.PUNCTUATION_CHARS):
        token_id = tokenizer.convert_tokens_to_ids(c)
        label_dict[token_id] = 1 + Constants.AUX_OFFSET + i
        # TODO: remove UNKs?
        print(
            f"auxiliary character {c} has token ID {token_id} and label {label_dict[token_id]}, decoded: {tokenizer.decode([token_id])}"
        )
        if token_id == tokenizer.unk_token_id:
            n_unks += 1

    print(f"found {n_unks} UNK tokens in auxiliary characters")

    # Map newline characters to token IDs with labels
    for c in label_args.newline_chars:
        token_id = tokenizer.convert_tokens_to_ids(c)
        label_dict[token_id] = 1 + Constants.NEWLINE_INDEX
        print(f"newline character {c} has token ID {token_id} and label {label_dict[token_id]}, decoded:")
        print(r"{}".format(tokenizer.decode([token_id])))

    return label_dict


tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})

label_dict = get_subword_label_dict(LabelArgs(), tokenizer)
print(len(label_dict))

def write_punctuation_file():
    with open(os.path.join(Constants.ROOT_DIR, "punctuation_xlmr.txt"), 'w', encoding='utf-8') as file:
        for char in Constants.PUNCTUATION_CHARS:
            token_id = tokenizer.convert_tokens_to_ids(char)
            if token_id != tokenizer.unk_token_id:
                file.write(char + '\n')
                
def write_punctuation_file_unk():
    added_unk = False
    with open(os.path.join(Constants.ROOT_DIR, "punctuation_xlmr_unk.txt"), 'w', encoding='utf-8') as file:
        for char in Constants.PUNCTUATION_CHARS:
            token_id = tokenizer.convert_tokens_to_ids(char)
            if token_id != tokenizer.unk_token_id:
                file.write(char + '\n')
            elif not added_unk:
                print("added unk")
                file.write('<unk>\n')
                added_unk = True

write_punctuation_file()
write_punctuation_file_unk()

label_args_default = LabelArgs()
print(Constants.PUNCTUATION_CHARS, len(Constants.PUNCTUATION_CHARS))

label_args_custom = LabelArgs(custom_punctuation_file='punctuation_xlmr.txt')
print(Constants.PUNCTUATION_CHARS, len(Constants.PUNCTUATION_CHARS))