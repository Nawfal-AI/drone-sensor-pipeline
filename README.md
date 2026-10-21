# drone-sensor-pipeline

End-to-end ML data engineering pipeline on VisDrone2019-DET.
Ingests drone images + annotations, generates synthetic GPS/IMU logs,
curates frames, and produces versioned WebDataset shards for model training.

## Project Structure

```
drone-sensor-pipeline/
├── data/
│   ├── raw/visdrone/
│   │   ├── VisDrone2019-DET-val/
│   │   │   ├── images/        ← your .jpg files
│   │   │   └── annotations/   ← your .txt files
│   │   └── VisDrone2019-DET-test-dev/
│   ├── processed/             ← auto-created by pipeline
│   └── sharded/               ← auto-created by pipeline
├── src/
│   ├── ingestion/
│   │   ├── visdrone_loader.py
│   │   ├── sensor_generator.py
│   │   └── sharder.py
│   ├── alignment/
│   │   └── temporal_aligner.py
│   ├── curation/
│   │   └── curator.py
│   └── dataloader/
│       └── drone_dataloader.py
├── scripts/
│   ├── run_ingestion.py
│   ├── run_alignment.py
│   ├── run_curation.py
│   ├── run_sharding.py
│   └── run_dataloader_benchmark.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── dvc.yaml
```

## Run without Docker (local)
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/run_ingestion.py --split val
python scripts/run_alignment.py
python scripts/run_curation.py
python scripts/run_sharding.py --shard-size 200
python scripts/run_dataloader_benchmark.py --num-workers 2 --num-batches 10
pytest tests/ -v
```

## Run with Docker
```bash
docker compose build pipeline
docker compose run pipeline
```
