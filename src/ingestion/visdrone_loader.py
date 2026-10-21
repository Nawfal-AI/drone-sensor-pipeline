"""
visdrone_loader.py
------------------
Loads VisDrone2019-DET images and bounding box annotations.

VisDrone annotation format (one .txt per image):
  x, y, width, height, score, category_id, truncation, occlusion

Categories:
  0=ignored, 1=pedestrian, 2=person, 3=car, 4=van,
  5=bus, 6=truck, 7=motor, 8=bicycle, 9=awning-tricycle, 10=tricycle

Output: data/processed/image_manifest.json
"""

import os
import cv2
import json
from pathlib import Path
from tqdm import tqdm

CATEGORY_NAMES = {
    0: "ignored", 1: "pedestrian", 2: "person", 3: "car", 4: "van",
    5: "bus", 6: "truck", 7: "motor", 8: "bicycle",
    9: "awning-tricycle", 10: "tricycle",
}


def parse_annotation_file(ann_path: str) -> list[dict]:
    annotations = []
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            score      = int(parts[4])
            cat_id     = int(parts[5])
            truncation = int(parts[6])
            occlusion  = int(parts[7])
            if cat_id == 0 or w == 0 or h == 0:
                continue
            annotations.append({
                "bbox":          [x, y, w, h],
                "score":         score,
                "category_id":   cat_id,
                "category_name": CATEGORY_NAMES.get(cat_id, "unknown"),
                "truncation":    truncation,
                "occlusion":     occlusion,
            })
    return annotations


def load_visdrone_split(
    split_dir: str,
    output_path: str = "data/processed/image_manifest.json",
    fps_equiv: float = 30.0,
) -> list[dict]:
    images_dir = Path(split_dir) / "images"
    ann_dir    = Path(split_dir) / "annotations"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotations folder not found: {ann_dir}")

    image_files = sorted(images_dir.glob("*.jpg"))
    if not image_files:
        image_files = sorted(images_dir.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    print(f"[VisDroneLoader] Found {len(image_files)} images in {images_dir}")

    manifest = []
    for idx, img_path in enumerate(tqdm(image_files, desc="Loading VisDrone images")):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: Cannot read {img_path} — skipping")
            continue
        h, w = img.shape[:2]
        timestamp_ms = (idx / fps_equiv) * 1000.0
        ann_file = ann_dir / (img_path.stem + ".txt")
        annotations = parse_annotation_file(str(ann_file)) if ann_file.exists() else []
        manifest.append({
            "image_id":     img_path.stem,
            "image_path":   str(img_path),
            "timestamp_ms": round(timestamp_ms, 3),
            "width":        w,
            "height":       h,
            "annotations":  annotations,
            "num_objects":  len(annotations),
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_boxes = sum(m["num_objects"] for m in manifest)
    print(f"[VisDroneLoader] {len(manifest)} images, {total_boxes} bounding boxes")
    print(f"[VisDroneLoader] Manifest → {output_path}")
    return manifest
