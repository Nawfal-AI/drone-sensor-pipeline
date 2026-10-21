"""Tests for curator."""
import sys, os, tempfile
import numpy as np, cv2, imagehash
from PIL import Image
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.curation.curator import laplacian_variance, compute_phash, is_duplicate

def _sharp(path):
    cv2.imwrite(path, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

def _blurry(path):
    img = cv2.GaussianBlur(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8), (99, 99), 0)
    cv2.imwrite(path, img)

def test_sharp_above_threshold():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f: path = f.name
    _sharp(path); score = laplacian_variance(path); os.unlink(path)
    assert score > 80

def test_blurry_below_threshold():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f: path = f.name
    _blurry(path); score = laplacian_variance(path); os.unlink(path)
    assert score < 80

def test_identical_is_duplicate():
    img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f: path = f.name
    img.save(path); h = compute_phash(path); os.unlink(path)
    assert is_duplicate(h, [h], 8)

def test_different_not_duplicate():
    hashes = []
    for _ in range(2):
        img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f: path = f.name
        img.save(path); hashes.append(compute_phash(path)); os.unlink(path)
    assert not is_duplicate(hashes[0], [hashes[1]], 8)
