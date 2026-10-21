"""Tests for temporal_aligner."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.alignment.temporal_aligner import _build_index, _nearest

RECORDS = [
    {"timestamp_ms": 0.0,   "lat": 39.9, "lon": 116.4, "alt": 80.0, "roll": 0.0, "pitch": 0.0, "yaw": 90.0},
    {"timestamp_ms": 20.0,  "lat": 39.9, "lon": 116.4, "alt": 80.1, "roll": 0.1, "pitch": 0.1, "yaw": 90.1},
    {"timestamp_ms": 40.0,  "lat": 39.9, "lon": 116.4, "alt": 80.2, "roll": 0.2, "pitch": 0.2, "yaw": 90.2},
    {"timestamp_ms": 100.0, "lat": 39.9, "lon": 116.4, "alt": 80.5, "roll": 0.5, "pitch": 0.5, "yaw": 91.0},
]

def test_exact_match():
    s, t = _build_index(RECORDS)
    assert _nearest(20.0, s, t, 500)["timestamp_ms"] == 20.0

def test_nearest_between_readings():
    s, t = _build_index(RECORDS)
    assert _nearest(25.0, s, t, 500)["timestamp_ms"] == 20.0

def test_gap_too_large_returns_none():
    s, t = _build_index(RECORDS)
    assert _nearest(2000.0, s, t, 500) is None

def test_within_gap_passes():
    s, t = _build_index(RECORDS)
    assert _nearest(699.0, s, t, 1000) is not None

def test_just_outside_gap():
    s, t = _build_index(RECORDS)
    assert _nearest(1101.0, s, t, 1000) is None
