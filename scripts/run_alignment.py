"""run_alignment.py — Step 2: align images to sensor readings."""
import argparse, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.alignment.temporal_aligner import align

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",   default="data/processed/image_manifest.json")
    p.add_argument("--sensor",     default="data/processed/sensor_records.json")
    p.add_argument("--output",     default="data/processed/aligned_metadata.json")
    p.add_argument("--max-gap-ms", type=float, default=1000.0)
    args = p.parse_args()
    print("=" * 60)
    print("  STEP 2 — Temporal Alignment")
    print("=" * 60)
    align(args.manifest, args.sensor, args.output, args.max_gap_ms)
    print("\n✓ Alignment complete")

if __name__ == "__main__":
    main()
