"""
sharder.py
----------
Packs curated VisDrone images into WebDataset .tar shards.
Each sample: <key>.jpg + <key>.json (metadata + annotations + GPS/IMU)
"""

import json
import os
import tarfile
from pathlib import Path
from tqdm import tqdm


def shard_dataset(
    curated_path: str = "data/processed/curated_metadata.json",
    output_dir: str   = "data/sharded",
    shard_size: int   = 200,
    shard_prefix: str = "shard",
) -> list[str]:
    with open(curated_path) as f:
        images = json.load(f)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    total_shards = (len(images) + shard_size - 1) // shard_size
    print(f"[Sharder] {len(images)} images → {total_shards} shards of ≤{shard_size}")

    shard_paths = []
    shard_index = []

    for shard_num in tqdm(range(total_shards), desc="Writing shards"):
        batch = images[shard_num * shard_size: (shard_num + 1) * shard_size]
        shard_filename = f"{shard_prefix}-{shard_num:06d}.tar"
        shard_path = os.path.join(output_dir, shard_filename)
        image_ids_in_shard = []

        with tarfile.open(shard_path, "w") as tar:
            for local_idx, img in enumerate(batch):
                key = f"{shard_num:06d}_{local_idx:06d}"
                tar.add(img["image_path"], arcname=f"{key}.jpg")
                meta = {
                    "image_id":         img["image_id"],
                    "timestamp_ms":     img["timestamp_ms"],
                    "width":            img["width"],
                    "height":           img["height"],
                    "num_objects":      img.get("num_objects", 0),
                    "annotations":      img.get("annotations", []),
                    "lat":              img.get("lat"),
                    "lon":              img.get("lon"),
                    "alt":              img.get("alt"),
                    "roll":             img.get("roll"),
                    "pitch":            img.get("pitch"),
                    "yaw":              img.get("yaw"),
                    "alignment_gap_ms": img.get("alignment_gap_ms"),
                    "laplacian_var":    img.get("laplacian_var"),
                }
                meta_tmp = f"/tmp/{key}.json"
                with open(meta_tmp, "w") as mf:
                    json.dump(meta, mf)
                tar.add(meta_tmp, arcname=f"{key}.json")
                os.remove(meta_tmp)
                image_ids_in_shard.append(img["image_id"])

        shard_paths.append(shard_path)
        shard_index.append({
            "shard_path":  shard_path,
            "num_samples": len(batch),
            "image_ids":   image_ids_in_shard,
        })

    index_path = os.path.join(output_dir, "shard_index.json")
    with open(index_path, "w") as f:
        json.dump(shard_index, f, indent=2)

    total_mb = sum(os.path.getsize(p) for p in shard_paths) / (1024 ** 2)
    print(f"[Sharder] Done — {total_mb:.1f} MB across {len(shard_paths)} shards")
    print(f"[Sharder] Index → {index_path}")
    return shard_paths
