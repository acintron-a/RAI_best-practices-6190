import os
import shutil
import pytest
from pyspark.sql import SparkSession
from src.ml_pipeline import run_ml_pipeline

@pytest.fixture(scope="module")
def spark_session():
    spark = SparkSession.builder \
        .appName("TestML") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()
    yield spark
    spark.stop()

def test_run_ml_pipeline(spark_session):
    # Setup test input and output
    input_path = "data/diabetes_sample_raw.csv"
    model_output_dir = "outputs/test_models/fair_diabetes_model"

    if os.path.exists(model_output_dir):
        shutil.rmtree(model_output_dir)

    # Run ML pipeline using sample data (which contains raw columns)
    run_ml_pipeline(input_path=input_path, model_output_dir=model_output_dir)

    # Verify model was persisted successfully
    assert os.path.exists(model_output_dir)
    assert os.path.isdir(model_output_dir)
