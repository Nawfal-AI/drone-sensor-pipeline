"""run_curation.py — Step 3: filter blurry, empty, duplicate images."""
import argparse, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.curation.curator import curate

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",          default="data/processed/aligned_metadata.json")
    p.add_argument("--output",         default="data/processed/curated_metadata.json")
    p.add_argument("--report",         default="data/processed/curation_report.json")
    p.add_argument("--blur-threshold", type=float, default=80.0)
    p.add_argument("--sim-threshold",  type=int,   default=8)
    p.add_argument("--keep-empty",     action="store_true")
    p.add_argument("--keep-unaligned", action="store_true")
    args = p.parse_args()
    print("=" * 60)
    print("  STEP 3 — Curation")
    print("=" * 60)
    curate(args.input, args.output, args.report,
           args.blur_threshold, args.sim_threshold,
           not args.keep_empty, not args.keep_unaligned)
    print("\n✓ Curation complete")

if __name__ == "__main__":
    main()
