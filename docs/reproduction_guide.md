# Reproduction Guide - Step-by-Step Execution

Follow these instructions to run and reproduce the batch pipeline, unit tests, and real-time streaming queries.

## Prerequisites

Ensure all dependencies listed in `requirements.txt` are installed:
```bash
pip install -r requirements.txt
```

## 1. Execute the Core Pipeline (Batch & Model Training)

To execute raw data ingestion, transformations, MLlib training, fairness evaluation, and the unit test suite, run the unified execution script:
```bash
./run.sh
```

This single command will:
1. Parse and ingest the raw dataset using explicit schemas (`src/ingestion.py`).
2. Apply transformations and perform Spark SQL aggregations (`src/transformations.py`).
3. Train the baseline and mitigated models, and save the serialized model directory to `outputs/models/fair_diabetes_model` (`src/ml_pipeline.py`).
4. Run the automated pytest suite (`tests/`).

## 2. Running the Real-Time Streaming Monitor

The streaming architecture simulates live clinical patient streams via TCP socket.

### Step A: Launch the Socket Stream Server
Open a new terminal window and start the clinical generator:
```bash
python src/patient_generator.py
```
This reads records sequentially from `data/diabetes_raw.csv` and broadcasts them on port `9999` at a rate of 2 records per second.

### Step B: Launch the Structured Streaming Client
Open another terminal window and run the real-time inference engine:
```bash
python src/streaming.py
```
This script will connect to the socket, apply the saved PipelineModel, and output real-time metric updates (Selection Rates and False Negative Rates by Race) to the console for every micro-batch. All predictions will be logged under `outputs/project_live_fairness_logs`.
