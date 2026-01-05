"""
Benchmark script to compare ONNX GPU runtime vs Triton server inference for SaT models.

Usage:
    # Make sure Triton server is running first:
    # sudo docker run --gpus=all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    #     -v /path/to/triton_models:/models \
    #     nvcr.io/nvidia/tritonserver:24.03-py3 \
    #     tritonserver --model-repository=/models

    python benchmark_inference.py \
        --model_name segment-any-text/sat-3l-sm \
        --triton_model_name sat_3l_sm \
        --triton_url localhost:8001
"""

import argparse
import time
import statistics
from typing import List, Tuple
import numpy as np
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# Sample texts of varying lengths for benchmarking
SAMPLE_TEXTS = {
    "short": "This is a short test. It has two sentences.",
    "medium": """Machine learning is a subset of artificial intelligence. It enables computers to learn from data. 
Deep learning is a subset of machine learning. Neural networks are the foundation of deep learning. 
These technologies have revolutionized many industries. Natural language processing is one application area.""",
    "long": """Artificial intelligence has transformed the way we interact with technology. From voice assistants 
to recommendation systems, AI is everywhere. Machine learning algorithms can identify patterns in vast amounts 
of data. This capability has led to breakthroughs in healthcare, finance, and transportation. Deep learning 
models can now generate human-like text and images. However, these advances also raise important ethical 
questions. How do we ensure AI systems are fair and unbiased? What happens when AI makes mistakes? These 
are questions society must grapple with. The future of AI holds both tremendous promise and significant 
challenges. Researchers continue to push the boundaries of what's possible. New architectures and training 
methods emerge regularly. The field moves at a breathtaking pace. Keeping up with the latest developments 
requires constant learning. But the potential rewards make it worthwhile. AI could help solve some of 
humanity's greatest challenges. Climate change, disease, and poverty might all be addressed with AI assistance.
The key is to develop these technologies responsibly. We must ensure they benefit everyone, not just a few.""",
}


def warmup(sat_model, text: str, num_warmup: int = 5):
    """Warm up the model to ensure accurate benchmarking."""
    for _ in range(num_warmup):
        sat_model.split(text)


def benchmark_inference(
    sat_model, 
    texts: List[str], 
    num_iterations: int = 100
) -> Tuple[List[float], List[int]]:
    """
    Benchmark inference time for a list of texts.
    
    Returns:
        Tuple of (latencies_ms, char_counts)
    """
    latencies = []
    char_counts = []
    
    for text in texts:
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            _ = sat_model.split(text)
            end_time = time.perf_counter()
            
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
            char_counts.append(len(text))
    
    return latencies, char_counts


def compute_statistics(latencies: List[float]) -> dict:
    """Compute statistics for latency measurements."""
    return {
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p90_ms": np.percentile(latencies, 90),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
    }


def print_results(name: str, stats: dict, total_chars: int, total_time_ms: float):
    """Print benchmark results in a formatted table."""
    throughput_chars_per_sec = (total_chars / total_time_ms) * 1000
    
    print(f"\n{'=' * 60}")
    print(f" {name} Results")
    print(f"{'=' * 60}")
    print(f"  Mean latency:      {stats['mean_ms']:>10.2f} ms")
    print(f"  Median latency:    {stats['median_ms']:>10.2f} ms")
    print(f"  Std deviation:     {stats['std_ms']:>10.2f} ms")
    print(f"  Min latency:       {stats['min_ms']:>10.2f} ms")
    print(f"  Max latency:       {stats['max_ms']:>10.2f} ms")
    print(f"  P50 latency:       {stats['p50_ms']:>10.2f} ms")
    print(f"  P90 latency:       {stats['p90_ms']:>10.2f} ms")
    print(f"  P95 latency:       {stats['p95_ms']:>10.2f} ms")
    print(f"  P99 latency:       {stats['p99_ms']:>10.2f} ms")
    print(f"  Throughput:        {throughput_chars_per_sec:>10.0f} chars/sec")
    print(f"{'=' * 60}")


def run_benchmark(
    model_name: str,
    triton_url: str = None,
    triton_model_name: str = None,
    num_iterations: int = 100,
    num_warmup: int = 10,
    text_sizes: List[str] = None,
):
    """Run the complete benchmark comparing ONNX and Triton inference."""
    from wtpsplit import SaT
    
    if text_sizes is None:
        text_sizes = ["short", "medium", "long"]
    
    texts = [SAMPLE_TEXTS[size] for size in text_sizes]
    
    results = {}
    
    # Benchmark ONNX GPU Runtime
    print("\n" + "=" * 60)
    print(" Benchmarking ONNX GPU Runtime")
    print("=" * 60)
    
    try:
        sat_onnx = SaT(
            model_name,
            ort_providers=["CUDAExecutionProvider"],
        )
        
        print("Warming up ONNX model...")
        warmup(sat_onnx, texts[0], num_warmup)
        
        print(f"Running {num_iterations} iterations per text...")
        onnx_latencies, onnx_chars = benchmark_inference(sat_onnx, texts, num_iterations)
        
        onnx_stats = compute_statistics(onnx_latencies)
        total_chars = sum(onnx_chars)
        total_time = sum(onnx_latencies)
        
        print_results("ONNX GPU Runtime", onnx_stats, total_chars, total_time)
        results["onnx"] = {
            "stats": onnx_stats,
            "total_chars": total_chars,
            "total_time_ms": total_time,
        }
        
        # Clean up
        del sat_onnx
        
    except Exception as e:
        print(f"ONNX benchmark failed: {e}")
        results["onnx"] = None
    
    # Benchmark Triton Server
    if triton_url and triton_model_name:
        print("\n" + "=" * 60)
        print(" Benchmarking Triton Server")
        print("=" * 60)
        
        try:
            sat_triton = SaT(
                model_name,
                triton_url=triton_url,
                triton_model_name=triton_model_name,
            )
            
            print("Warming up Triton model...")
            warmup(sat_triton, texts[0], num_warmup)
            
            print(f"Running {num_iterations} iterations per text...")
            triton_latencies, triton_chars = benchmark_inference(sat_triton, texts, num_iterations)
            
            triton_stats = compute_statistics(triton_latencies)
            total_chars = sum(triton_chars)
            total_time = sum(triton_latencies)
            
            print_results("Triton Server", triton_stats, total_chars, total_time)
            results["triton"] = {
                "stats": triton_stats,
                "total_chars": total_chars,
                "total_time_ms": total_time,
            }
            
            # Clean up
            del sat_triton
            
        except Exception as e:
            print(f"Triton benchmark failed: {e}")
            results["triton"] = None
    
    # Compare results
    if results.get("onnx") and results.get("triton"):
        print("\n" + "=" * 60)
        print(" Comparison Summary")
        print("=" * 60)
        
        onnx_mean = results["onnx"]["stats"]["mean_ms"]
        triton_mean = results["triton"]["stats"]["mean_ms"]
        
        speedup = onnx_mean / triton_mean if triton_mean > 0 else 0
        
        print(f"  ONNX mean latency:   {onnx_mean:>10.2f} ms")
        print(f"  Triton mean latency: {triton_mean:>10.2f} ms")
        print(f"  Speedup (Triton vs ONNX): {speedup:>6.2f}x")
        
        if speedup > 1:
            print(f"  → Triton is {speedup:.2f}x faster than ONNX")
        else:
            print(f"  → ONNX is {1/speedup:.2f}x faster than Triton")
        
        # Throughput comparison
        onnx_throughput = (results["onnx"]["total_chars"] / results["onnx"]["total_time_ms"]) * 1000
        triton_throughput = (results["triton"]["total_chars"] / results["triton"]["total_time_ms"]) * 1000
        
        print(f"\n  ONNX throughput:   {onnx_throughput:>10.0f} chars/sec")
        print(f"  Triton throughput: {triton_throughput:>10.0f} chars/sec")
        print("=" * 60)
    
    return results


def run_batch_benchmark(
    model_name: str,
    triton_url: str = None,
    triton_model_name: str = None,
    batch_sizes: List[int] = None,
    num_iterations: int = 50,
):
    """Benchmark with different batch sizes (multiple texts at once)."""
    from wtpsplit import SaT
    
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32]
    
    base_text = SAMPLE_TEXTS["medium"]
    
    print("\n" + "=" * 60)
    print(" Batch Size Benchmark")
    print("=" * 60)
    
    results = {"onnx": {}, "triton": {}}
    
    # ONNX benchmark
    try:
        sat_onnx = SaT(model_name, ort_providers=["CUDAExecutionProvider"])
        
        print("\nONNX GPU Runtime - Batch benchmarks:")
        for batch_size in batch_sizes:
            batch_texts = [base_text] * batch_size
            total_chars = batch_size * len(base_text)
            
            # Warmup
            for _ in range(5):
                # Force evaluation by converting to list
                _ = list(sat_onnx.split(batch_texts))
            
            latencies = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                # Force evaluation by converting to list
                _ = list(sat_onnx.split(batch_texts))
                end = time.perf_counter()
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
            
            mean_latency = statistics.mean(latencies)
            std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
            throughput = (total_chars / mean_latency) * 1000 if mean_latency > 0 else 0
            
            print(f"  Batch size {batch_size:>3}: {mean_latency:>8.2f} ms ± {std_latency:>6.2f}, {throughput:>10.0f} chars/sec")
            results["onnx"][batch_size] = {"mean_ms": mean_latency, "std_ms": std_latency, "throughput": throughput}
        
        del sat_onnx
    except Exception as e:
        print(f"ONNX batch benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Triton benchmark
    if triton_url and triton_model_name:
        try:
            sat_triton = SaT(model_name, triton_url=triton_url, triton_model_name=triton_model_name)
            
            print("\nTriton Server - Batch benchmarks:")
            for batch_size in batch_sizes:
                batch_texts = [base_text] * batch_size
                total_chars = batch_size * len(base_text)
                
                # Warmup
                for _ in range(5):
                    # Force evaluation by converting to list
                    _ = list(sat_triton.split(batch_texts))
                
                latencies = []
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    # Force evaluation by converting to list
                    _ = list(sat_triton.split(batch_texts))
                    end = time.perf_counter()
                    latency_ms = (end - start) * 1000
                    latencies.append(latency_ms)
                
                mean_latency = statistics.mean(latencies)
                std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
                throughput = (total_chars / mean_latency) * 1000 if mean_latency > 0 else 0
                
                print(f"  Batch size {batch_size:>3}: {mean_latency:>8.2f} ms ± {std_latency:>6.2f}, {throughput:>10.0f} chars/sec")
                results["triton"][batch_size] = {"mean_ms": mean_latency, "std_ms": std_latency, "throughput": throughput}
            
            del sat_triton
        except Exception as e:
            print(f"Triton batch benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def run_concurrent_benchmark(
    model_name: str,
    triton_url: str = None,
    triton_model_name: str = None,
    num_clients: List[int] = None,
    requests_per_client: int = 20,
    skip_onnx: bool = False,
):
    """
    Benchmark with multiple concurrent clients to test throughput under load.
    This is where Triton should shine due to its dynamic batching.
    """
    from wtpsplit import SaT
    
    if num_clients is None:
        num_clients = [1, 2, 4, 8, 16, 32]
    
    base_text = SAMPLE_TEXTS["medium"]
    total_chars_per_request = len(base_text)
    
    print("\n" + "=" * 70)
    print(" Concurrent Clients Benchmark (simulates production load)")
    print("=" * 70)
    print(f"  Requests per client: {requests_per_client}")
    print(f"  Text length: {total_chars_per_request} chars")
    print("=" * 70)
    
    results = {"onnx": {}, "triton": {}}
    
    def client_worker(sat_model, text: str, num_requests: int) -> List[float]:
        """Worker function for a single client."""
        latencies = []
        for _ in range(num_requests):
            start = time.perf_counter()
            _ = sat_model.split(text)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        return latencies
    
    # ONNX benchmark
    if not skip_onnx:
        try:
            sat_onnx = SaT(model_name, ort_providers=["CUDAExecutionProvider"])
            
            # Warmup
            for _ in range(10):
                _ = sat_onnx.split(base_text)
            
            print("\nONNX GPU Runtime - Concurrent clients:")
            print(f"  {'Clients':>8} | {'Total Time':>12} | {'Throughput':>15} | {'Avg Latency':>12} | {'P99 Latency':>12}")
            print("  " + "-" * 68)
            
            for n_clients in num_clients:
                all_latencies = []
                
                # Run concurrent clients
                start_time = time.perf_counter()
                with ThreadPoolExecutor(max_workers=n_clients) as executor:
                    futures = [
                        executor.submit(client_worker, sat_onnx, base_text, requests_per_client)
                        for _ in range(n_clients)
                    ]
                    for future in as_completed(futures):
                        all_latencies.extend(future.result())
                end_time = time.perf_counter()
                
                total_time_sec = end_time - start_time
                total_requests = n_clients * requests_per_client
                total_chars = total_requests * total_chars_per_request
                throughput = total_chars / total_time_sec
                avg_latency = statistics.mean(all_latencies)
                p99_latency = np.percentile(all_latencies, 99)
                
                print(f"  {n_clients:>8} | {total_time_sec:>10.2f} s | {throughput:>12.0f} c/s | {avg_latency:>10.2f} ms | {p99_latency:>10.2f} ms")
                results["onnx"][n_clients] = {
                    "total_time_sec": total_time_sec,
                    "throughput_chars_sec": throughput,
                    "avg_latency_ms": avg_latency,
                    "p99_latency_ms": p99_latency,
                }
            
            del sat_onnx
        except Exception as e:
            print(f"ONNX concurrent benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Triton benchmark
    if triton_url and triton_model_name:
        try:
            sat_triton = SaT(model_name, triton_url=triton_url, triton_model_name=triton_model_name)
            
            # Warmup
            for _ in range(10):
                _ = sat_triton.split(base_text)
            
            print("\nTriton Server - Concurrent clients:")
            print(f"  {'Clients':>8} | {'Total Time':>12} | {'Throughput':>15} | {'Avg Latency':>12} | {'P99 Latency':>12}")
            print("  " + "-" * 68)
            
            for n_clients in num_clients:
                all_latencies = []
                
                # Run concurrent clients
                start_time = time.perf_counter()
                with ThreadPoolExecutor(max_workers=n_clients) as executor:
                    futures = [
                        executor.submit(client_worker, sat_triton, base_text, requests_per_client)
                        for _ in range(n_clients)
                    ]
                    for future in as_completed(futures):
                        all_latencies.extend(future.result())
                end_time = time.perf_counter()
                
                total_time_sec = end_time - start_time
                total_requests = n_clients * requests_per_client
                total_chars = total_requests * total_chars_per_request
                throughput = total_chars / total_time_sec
                avg_latency = statistics.mean(all_latencies)
                p99_latency = np.percentile(all_latencies, 99)
                
                print(f"  {n_clients:>8} | {total_time_sec:>10.2f} s | {throughput:>12.0f} c/s | {avg_latency:>10.2f} ms | {p99_latency:>10.2f} ms")
                results["triton"][n_clients] = {
                    "total_time_sec": total_time_sec,
                    "throughput_chars_sec": throughput,
                    "avg_latency_ms": avg_latency,
                    "p99_latency_ms": p99_latency,
                }
            
            del sat_triton
        except Exception as e:
            print(f"Triton concurrent benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Comparison
    if results.get("onnx") and results.get("triton"):
        print("\n" + "=" * 70)
        print(" Concurrent Benchmark Comparison (Triton speedup vs ONNX)")
        print("=" * 70)
        print(f"  {'Clients':>8} | {'ONNX Throughput':>18} | {'Triton Throughput':>18} | {'Speedup':>10}")
        print("  " + "-" * 62)
        
        for n_clients in num_clients:
            if n_clients in results["onnx"] and n_clients in results["triton"]:
                onnx_tp = results["onnx"][n_clients]["throughput_chars_sec"]
                triton_tp = results["triton"][n_clients]["throughput_chars_sec"]
                speedup = triton_tp / onnx_tp if onnx_tp > 0 else 0
                
                print(f"  {n_clients:>8} | {onnx_tp:>15.0f} c/s | {triton_tp:>15.0f} c/s | {speedup:>9.2f}x")
        
        print("=" * 70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ONNX vs Triton inference for SaT models")
    parser.add_argument(
        "--model_name",
        type=str,
        default="segment-any-text/sat-3l-sm",
        help="Model name or path",
    )
    parser.add_argument(
        "--triton_url",
        type=str,
        default="localhost:8001",
        help="Triton server URL",
    )
    parser.add_argument(
        "--triton_model_name",
        type=str,
        default="sat_3l_sm",
        help="Triton model name",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of iterations per benchmark",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--skip_onnx",
        action="store_true",
        help="Skip ONNX benchmark",
    )
    parser.add_argument(
        "--skip_triton",
        action="store_true",
        help="Skip Triton benchmark",
    )
    parser.add_argument(
        "--batch_benchmark",
        action="store_true",
        help="Run batch size benchmark",
    )
    parser.add_argument(
        "--concurrent_benchmark",
        action="store_true",
        help="Run concurrent clients benchmark (simulates production load)",
    )
    parser.add_argument(
        "--requests_per_client",
        type=int,
        default=20,
        help="Number of requests per client in concurrent benchmark",
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print(" SaT Inference Benchmark: ONNX vs Triton")
    print("=" * 60)
    print(f"  Model: {args.model_name}")
    print(f"  Triton URL: {args.triton_url}")
    print(f"  Triton Model: {args.triton_model_name}")
    print(f"  Iterations: {args.num_iterations}")
    print(f"  Warmup: {args.num_warmup}")
    print("=" * 60)
    
    # Run main benchmark
    run_benchmark(
        model_name=args.model_name,
        triton_url=None if args.skip_triton else args.triton_url,
        triton_model_name=None if args.skip_triton else args.triton_model_name,
        num_iterations=args.num_iterations,
        num_warmup=args.num_warmup,
    )
    
    # Run batch benchmark if requested
    if args.batch_benchmark:
        run_batch_benchmark(
            model_name=args.model_name,
            triton_url=None if args.skip_triton else args.triton_url,
            triton_model_name=None if args.skip_triton else args.triton_model_name,
        )
    
    # Run concurrent benchmark if requested
    if args.concurrent_benchmark:
        run_concurrent_benchmark(
            model_name=args.model_name,
            triton_url=None if args.skip_triton else args.triton_url,
            triton_model_name=None if args.skip_triton else args.triton_model_name,
            requests_per_client=args.requests_per_client,
            skip_onnx=args.skip_onnx,
        )
    
    print("\nBenchmark complete!")
