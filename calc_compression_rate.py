from datasets import load_dataset
from transformers import XLMRobertaTokenizer

def calculate_compression_rate(dataset_name):
    # Load the dataset
    dataset = load_dataset(dataset_name, split='train')

    # Initialize the tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

    total_chars = 0
    total_tokens = 0

    # Iterate over the dataset
    for sample in dataset:
        text = sample['text']
        total_chars += len(text)

        # Tokenize the text
        tokens = tokenizer.tokenize(text)
        total_tokens += len(tokens)

    # Calculate the average compression rate
    avg_compression_rate = total_chars / total_tokens if total_tokens > 0 else 0

    return avg_compression_rate

# Example dataset
dataset_name = "markus583/mC4-TEST"
compression_rate = calculate_compression_rate(dataset_name)
print(compression_rate)
