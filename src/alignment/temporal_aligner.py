"""
temporal_aligner.py
--------------------
Aligns each VisDrone image to its nearest sensor reading by timestamp.
Uses binary search (bisect) for fast O(log n) lookup.
"""

import json
import bisect
from pathlib import Path
from tqdm import tqdm


def _build_index(sensor_records):
    sorted_records = sorted(sensor_records, key=lambda r: r["timestamp_ms"])
    timestamps = [r["timestamp_ms"] for r in sorted_records]
    return sorted_records, timestamps


def _nearest(frame_ts, sorted_records, timestamps, max_gap_ms):
    if not timestamps:
        return None
    pos = bisect.bisect_left(timestamps, frame_ts)
    candidates = []
    if pos < len(timestamps):
        candidates.append(sorted_records[pos])
    if pos > 0:
        candidates.append(sorted_records[pos - 1])
    best = min(candidates, key=lambda r: abs(r["timestamp_ms"] - frame_ts))
    if abs(best["timestamp_ms"] - frame_ts) > max_gap_ms:
        return None
    return best


def align(
    manifest_path: str = "data/processed/image_manifest.json",
    sensor_path: str   = "data/processed/sensor_records.json",
    output_path: str   = "data/processed/aligned_metadata.json",
    max_gap_ms: float  = 1000.0,
) -> list[dict]:
    with open(manifest_path) as f:
        images = json.load(f)
    with open(sensor_path) as f:
        sensor_records = json.load(f)

    sorted_records, timestamps = _build_index(sensor_records)
    aligned = []
    unaligned = 0

    for img in tqdm(images, desc="Aligning images to sensor"):
        ts = img["timestamp_ms"]
        nearest = _nearest(ts, sorted_records, timestamps, max_gap_ms)
        entry = {**img, "aligned": nearest is not None}
        if nearest:
            entry.update({
                "lat":                 nearest["lat"],
                "lon":                 nearest["lon"],
                "alt":                 nearest["alt"],
                "roll":                nearest["roll"],
                "pitch":               nearest["pitch"],
                "yaw":                 nearest["yaw"],
                "sensor_timestamp_ms": nearest["timestamp_ms"],
                "alignment_gap_ms":    round(abs(nearest["timestamp_ms"] - ts), 3),
            })
        else:
            entry.update({
                "lat": None, "lon": None, "alt": None,
                "roll": None, "pitch": None, "yaw": None,
                "sensor_timestamp_ms": None, "alignment_gap_ms": None,
            })
            unaligned += 1
        aligned.append(entry)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(aligned, f, indent=2)

    print(f"[Aligner] Total: {len(images)}  "
          f"Aligned: {len(images)-unaligned}  Unaligned: {unaligned}")
    print(f"[Aligner] Output → {output_path}")
    return aligned
