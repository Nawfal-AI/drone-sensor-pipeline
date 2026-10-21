"""
drone_dataloader.py
--------------------
PyTorch DataLoader backed by VisDrone WebDataset shards.
Returns (image_tensor, metadata_dict) batches.
"""

import io, json, glob, time, warnings
import torch
import webdataset as wds
import torchvision.transforms as T
from PIL import Image

# Suppress NumPy 1.x/2.x compatibility warning from torch internals
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

IMAGENET_TRANSFORMS = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def decode_sample(sample):
    img  = Image.open(io.BytesIO(sample["jpg"])).convert("RGB")
    meta = json.loads(sample["json"].decode("utf-8"))
    return img, meta


def collate_fn(batch):
    images, metas = zip(*batch)
    return torch.stack(images), list(metas)


def build_dataloader(
    shard_dir: str,
    batch_size: int     = 16,
    num_workers: int    = 4,
    shuffle_buffer: int = 200,
    transforms          = None,
    split: str          = "train",
) -> wds.WebLoader:
    transforms = transforms or IMAGENET_TRANSFORMS
    shard_paths = sorted(glob.glob(f"{shard_dir}/*.tar"))
    if not shard_paths:
        raise FileNotFoundError(f"No .tar shards found in: {shard_dir}")
    print(f"[DataLoader] {len(shard_paths)} shards  "
          f"batch_size={batch_size}  workers={num_workers}  split={split}")

    def process(sample):
        img, meta = decode_sample(sample)
        return transforms(img), meta

    dataset = (
        wds.WebDataset(shard_paths, resampled=(split == "train"),
                       shardshuffle=(split == "train"))   # fixes shardshuffle warning
        .shuffle(shuffle_buffer if split == "train" else 0)
        .map(process)
        .batched(batch_size, collation_fn=collate_fn)
    )
    return wds.WebLoader(
        dataset, batch_size=None, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def benchmark_throughput(
    shard_dir: str   = "data/sharded",
    batch_size: int  = 16,
    num_workers: int = 4,
    num_batches: int = 20,
) -> dict:
    loader = build_dataloader(shard_dir, batch_size=batch_size,
                               num_workers=num_workers, split="train")
    loader_iter = iter(loader)
    print("[Benchmark] Warming up...")
    for _ in range(3):
        try: next(loader_iter)
        except StopIteration: break

    print(f"[Benchmark] Timing {num_batches} batches...")
    start = time.perf_counter()
    samples_seen = 0
    for i, (images, metas) in enumerate(loader_iter):
        samples_seen += images.shape[0]
        if i == 0:
            print(f"\n[Benchmark] First batch:")
            print(f"  Image tensor shape : {images.shape}")
            print(f"  image_id={metas[0].get('image_id')}  "
                  f"objects={metas[0].get('num_objects')}  "
                  f"lat={metas[0].get('lat')}  "
                  f"alt={metas[0].get('alt')}m")
        if i + 1 >= num_batches:
            break

    elapsed = time.perf_counter() - start
    result = {
        "samples_seen":    samples_seen,
        "elapsed_sec":     round(elapsed, 2),
        "samples_per_sec": round(samples_seen / elapsed, 1),
        "batches_per_sec": round(num_batches / elapsed, 2),
        "batch_size":      batch_size,
        "num_workers":     num_workers,
    }
    print("\n[Benchmark] Throughput:")
    for k, v in result.items():
        print(f"  {k:20s}: {v}")
    return result