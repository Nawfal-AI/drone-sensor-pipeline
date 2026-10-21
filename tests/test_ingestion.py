"""Tests for visdrone_loader and sensor_generator."""
import sys, os, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.ingestion.sensor_generator import generate_sensor_log
from src.ingestion.visdrone_loader import parse_annotation_file

def test_sensor_log_length():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f: path = f.name
    records = generate_sensor_log(1000.0, 50.0, path)
    os.unlink(path)
    assert len(records) >= 50

def test_sensor_log_sorted():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f: path = f.name
    records = generate_sensor_log(500.0, output_path=path)
    os.unlink(path)
    ts = [r["timestamp_ms"] for r in records]
    assert ts == sorted(ts)

def test_sensor_has_required_fields():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f: path = f.name
    records = generate_sensor_log(100.0, output_path=path)
    os.unlink(path)
    required = {"timestamp_ms", "lat", "lon", "alt", "roll", "pitch", "yaw"}
    for r in records:
        assert required.issubset(r.keys())

def test_sensor_altitude_in_range():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f: path = f.name
    records = generate_sensor_log(2000.0, output_path=path)
    os.unlink(path)
    for r in records:
        assert 50.0 <= r["alt"] <= 120.0

def test_parse_valid_annotation():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("100,200,50,80,1,3,0,0\n"); path = f.name
    anns = parse_annotation_file(path); os.unlink(path)
    assert len(anns) == 1
    assert anns[0]["category_name"] == "car"
    assert anns[0]["bbox"] == [100, 200, 50, 80]

def test_parse_ignores_category_zero():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("0,0,100,100,0,0,0,0\n"); path = f.name
    anns = parse_annotation_file(path); os.unlink(path)
    assert len(anns) == 0

def test_parse_ignores_zero_area():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("10,20,0,50,1,1,0,0\n"); path = f.name
    anns = parse_annotation_file(path); os.unlink(path)
    assert len(anns) == 0

def test_parse_multiple_annotations():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("10,20,30,40,1,1,0,0\n50,60,70,80,1,3,0,1\n0,0,10,10,0,0,0,0\n")
        path = f.name
    anns = parse_annotation_file(path); os.unlink(path)
    assert len(anns) == 2
    assert {a["category_name"] for a in anns} == {"pedestrian", "car"}
