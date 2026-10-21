"""run_dataloader_benchmark.py — Step 5: benchmark PyTorch DataLoader throughput."""
import argparse, json, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataloader.drone_dataloader import benchmark_throughput

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard-dir",   default="data/sharded")
    p.add_argument("--batch-size",  type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-batches", type=int, default=20)
    p.add_argument("--output",      default="benchmark_results.json")
    args = p.parse_args()
    print("=" * 60)
    print("  STEP 5 — DataLoader Benchmark")
    print("=" * 60)
    result = benchmark_throughput(args.shard_dir, args.batch_size,
                                   args.num_workers, args.num_batches)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Results saved → {args.output}")

if __name__ == "__main__":
    main()
