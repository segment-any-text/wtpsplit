import nnsplit
import json
import time

if __name__ == "__main__":
    data = json.load(open("../../benchmarks/sample.json", "r"))

    for batch_size in [256, 1024]:
        for use_cuda in [False, True]:
            model = nnsplit.NNSplit.load("de", use_cuda=use_cuda, batch_size=batch_size)

            print(f"{batch_size} {'GPU' if use_cuda else 'CPU'}")
            start = time.time()

            model.split(data)

            print(f"Time: {time.time() - start}")
