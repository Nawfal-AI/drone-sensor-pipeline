"""run_sharding.py — Step 4: pack curated images into WebDataset .tar shards."""
import argparse, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.ingestion.sharder import shard_dataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      default="data/processed/curated_metadata.json")
    p.add_argument("--output-dir", default="data/sharded")
    p.add_argument("--shard-size", type=int, default=200)
    args = p.parse_args()
    print("=" * 60)
    print("  STEP 4 — Sharding")
    print("=" * 60)
    shards = shard_dataset(args.input, args.output_dir, args.shard_size)
    print(f"\n✓ Sharding complete — {len(shards)} shard(s)")

if __name__ == "__main__":
    main()
