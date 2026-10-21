"""run_ingestion.py — Step 1: load images + annotations + generate sensor log."""
import argparse, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.ingestion.visdrone_loader import load_visdrone_split
from src.ingestion.sensor_generator import generate_sensor_log

SPLIT_DIRS = {
    "val":      "data/raw/visdrone/VisDrone2019-DET-val",
    "test-dev": "data/raw/visdrone/VisDrone2019-DET-test-dev",
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split",        default="val", choices=["val", "test-dev"])
    p.add_argument("--data-dir",     default=None)
    p.add_argument("--manifest-out", default="data/processed/image_manifest.json")
    p.add_argument("--sensor-out",   default="data/processed/sensor_records.json")
    p.add_argument("--fps-equiv",    type=float, default=30.0)
    p.add_argument("--sensor-hz",    type=float, default=50.0)
    args = p.parse_args()

    split_dir = args.data_dir or SPLIT_DIRS[args.split]
    print("=" * 60)
    print("  STEP 1 — Ingestion")
    print("=" * 60)
    print(f"  Split : {split_dir}")

    print("\n[1/2] Loading VisDrone images and annotations...")
    manifest = load_visdrone_split(split_dir, args.manifest_out, args.fps_equiv)

    print("\n[2/2] Generating synthetic GPS/IMU sensor log...")
    duration_ms = manifest[-1]["timestamp_ms"] + 2000.0
    generate_sensor_log(duration_ms, args.sensor_hz, args.sensor_out)

    print(f"\n✓ Ingestion complete — {len(manifest)} images")

if __name__ == "__main__":
    main()
