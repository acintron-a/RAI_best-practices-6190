#!/bin/bash
# -------------------------------------------------------------------------
# Unified execution script for the Spark Data Analysis Project
# -------------------------------------------------------------------------
set -e

echo "======================================================="
echo " STARTING COHORT DATA PIPELINE AND ML TRAINING"
echo "======================================================="

# Step 1: Run Ingestion & SQL EDA
echo -e "\n[Step 1/4] Running raw data ingestion..."
python src/ingestion.py

# Step 2: Run transformations & clinical profiling
echo -e "\n[Step 2/4] Running data transformations..."
python src/transformations.py

# Step 3: Run offline MLlib model training and fairness auditing
echo -e "\n[Step 3/4] Running offline model training & bias mitigation..."
python src/ml_pipeline.py

# Step 4: Execute automated unit test suite
echo -e "\n[Step 4/4] Executing test suite via pytest..."
python -m pytest

echo -e "\n======================================================="
echo " PIPELINE RUN AND VERIFICATION SUCCESSFULLY COMPLETED!"
echo "======================================================="
echo "To execute real-time streaming inference:"
echo "1. Run: python src/patient_generator.py (in a separate terminal)"
echo "2. Run: python src/streaming.py"
echo "======================================================="
