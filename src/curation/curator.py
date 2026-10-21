"""
curator.py
----------
Curates VisDrone images:
  1. Drop images with no annotations (empty scenes)
  2. Drop blurry images (Laplacian variance)
  3. Drop near-duplicate images (perceptual hash)
"""

import json
import cv2
import imagehash
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def laplacian_variance(image_path: str) -> float:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def compute_phash(image_path: str) -> imagehash.ImageHash:
    return imagehash.phash(Image.open(image_path).convert("RGB"))


def is_duplicate(new_hash, seen_hashes, threshold=8) -> bool:
    return any((new_hash - seen) <= threshold for seen in seen_hashes)


def curate(
    aligned_path: str     = "data/processed/aligned_metadata.json",
    curated_path: str     = "data/processed/curated_metadata.json",
    report_path: str      = "data/processed/curation_report.json",
    blur_threshold: float = 80.0,
    hash_threshold: int   = 8,
    drop_empty: bool      = True,
    skip_unaligned: bool  = True,
) -> list[dict]:
    with open(aligned_path) as f:
        images = json.load(f)

    curated = []
    report  = []
    seen_hashes = []
    stats = {
        "total": len(images), "passed": 0,
        "dropped_unaligned": 0, "dropped_empty": 0,
        "dropped_blur": 0, "dropped_duplicate": 0,
    }

    for img in tqdm(images, desc="Curating images"):
        entry = {"image_id": img["image_id"], "image_path": img["image_path"]}

        if skip_unaligned and not img.get("aligned", True):
            entry.update({"status": "dropped", "reason": "unaligned"})
            report.append(entry); stats["dropped_unaligned"] += 1; continue

        if drop_empty and img.get("num_objects", 0) == 0:
            entry.update({"status": "dropped", "reason": "no_annotations"})
            report.append(entry); stats["dropped_empty"] += 1; continue

        lap = laplacian_variance(img["image_path"])
        if lap < blur_threshold:
            entry.update({"status": "dropped", "reason": f"blur(score={lap:.1f})"})
            report.append(entry); stats["dropped_blur"] += 1; continue

        ph = compute_phash(img["image_path"])
        if is_duplicate(ph, seen_hashes, hash_threshold):
            entry.update({"status": "dropped", "reason": "duplicate"})
            report.append(entry); stats["dropped_duplicate"] += 1; continue

        seen_hashes.append(ph)
        img["laplacian_var"] = round(lap, 2)
        curated.append(img)
        entry.update({"status": "passed", "laplacian_var": round(lap, 2),
                       "num_objects": img.get("num_objects", 0)})
        report.append(entry)
        stats["passed"] += 1

    for path, data in [(curated_path, curated), (report_path, report)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    print("\n[Curator] Results:")
    for k, v in stats.items():
        print(f"  {k:28s}: {v}")
    print(f"  Output → {curated_path}")
    return curated
