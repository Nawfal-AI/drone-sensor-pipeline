"""
sensor_generator.py
--------------------
Generates a synthetic GPS/IMU sensor log for the image sequence.
VisDrone has no real GPS/IMU so we simulate a realistic drone flight.
"""

import json
import random
import math
from pathlib import Path


def generate_sensor_log(
    duration_ms: float,
    sample_rate_hz: float = 50.0,
    output_path: str = "data/processed/sensor_records.json",
    start_lat: float = 39.9042,
    start_lon: float = 116.4074,
    start_alt: float = 80.0,
    seed: int = 42,
) -> list[dict]:
    random.seed(seed)
    dt_ms = 1000.0 / sample_rate_hz
    num_samples = int(math.ceil(duration_ms / dt_ms)) + 10

    records = []
    lat, lon, alt = start_lat, start_lon, start_alt
    yaw = 90.0

    for i in range(num_samples):
        ts = i * dt_ms
        lat += 0.000004 + random.gauss(0, 0.0000005)
        lon += 0.000003 + random.gauss(0, 0.0000005)
        alt += random.gauss(0, 0.03)
        alt  = max(50.0, min(120.0, alt))
        roll  = random.gauss(0.0, 0.6)
        pitch = random.gauss(-2.0, 0.4)
        yaw  += random.gauss(0.0, 0.05)
        records.append({
            "timestamp_ms": round(ts, 3),
            "lat":   round(lat, 7),
            "lon":   round(lon, 7),
            "alt":   round(alt, 2),
            "roll":  round(roll, 4),
            "pitch": round(pitch, 4),
            "yaw":   round(yaw % 360, 4),
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"[SensorGenerator] {len(records)} records @ {sample_rate_hz}Hz "
          f"covering {duration_ms/1000:.1f}s → {output_path}")
    return records
