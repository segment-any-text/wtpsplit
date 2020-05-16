import spacy
import json
import time

spacy.require_gpu()

if __name__ == "__main__":
    data = json.load(open("benchmarks/sample.json", "r"))
    
    for batch_size in [256, 1024]:
        nlp = spacy.load("de_core_news_sm", disable=["tagger", "parser", "ner"])

        print(f"{batch_size}")
        start = time.time()

        for doc in nlp.pipe(data, batch_size=batch_size):
            continue

        print(f"Time: {time.time() - start}")
        