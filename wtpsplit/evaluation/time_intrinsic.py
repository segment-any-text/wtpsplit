import argparse
import sys
import pandas as pd
import math
from multiprocessing import Process, Queue
import intrinsic


def run_intrinsic_with_stride(stride, args, results_queue):
    modified_args = argparse.Namespace(**vars(args))
    modified_args.stride = stride
    results, results_avg, total_test_time = intrinsic.main(modified_args)  # Capture results
    results_queue.put((stride, results, results_avg, total_test_time))


def benchmark_strides(low_stride, high_stride, args):
    stride_values = [2**i for i in range(int(math.log2(low_stride)), int(math.log2(high_stride)) + 1)]
    results_data = []

    for stride in stride_values:
        results_queue = Queue()
        p = Process(target=run_intrinsic_with_stride, args=(stride, args, results_queue))
        p.start()
        p.join()

        # intrinsic.main() returns a tuple of (results, results_avg, total_test_timee)
        stride, stride_results, stride_results_avg, total_test_time = results_queue.get()

        results_data.append(
            {
                "stride": stride,
                "block_size": args.block_size,
                "batch_size": args.batch_size,
                "execution_time": total_test_time,
                "results": stride_results,
                "results_avg": stride_results_avg,
                "threshold": args.threshold,
                "include_langs": args.include_langs,
                "max_n_train_sentences": args.max_n_train_sentences,
            }
        )
        print(results_data)

    return pd.DataFrame(results_data)


if __name__ == "__main__":
    # Extract low_stride and high_stride values
    stride_args = ["--low_stride", "--high_stride"]
    strides = {}

    # Iterate over stride_args to extract and remove them from sys.argv
    for stride_arg in stride_args:
        if stride_arg in sys.argv:
            index = sys.argv.index(stride_arg)
            try:
                strides[stride_arg] = int(sys.argv[index + 1])
                # Remove the stride argument and its value
                del sys.argv[index : index + 2]
            except (IndexError, ValueError):
                raise ValueError(f"Invalid or missing value for {stride_arg}.")

    if "--low_stride" not in strides or "--high_stride" not in strides:
        raise ValueError("Both --low_stride and --high_stride must be provided.")

    low_stride = strides["--low_stride"]
    high_stride = strides["--high_stride"]

    # Remaining arguments are passed to intrinsic.Args
    args = intrinsic.HfArgumentParser(intrinsic.Args).parse_args_into_dataclasses()[0]

    df_results = benchmark_strides(low_stride, high_stride, args)
    print(df_results)
    # Optionally save df_results to a file
    # to csv
    df_results.to_csv(
        f"timing_results_{args.model_path.replace('/','__')}_batch{args.batch_size}_b{args.block_size}+s{args.stride}_n{args.max_n_train_sentences}_u{args.threshold}_AVG.csv"
    )
