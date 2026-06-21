import os
import pytest
from pyspark.sql import SparkSession
from src.ingestion import run_ingestion

@pytest.fixture(scope="module")
def spark_session():
    # Setup test Spark session
    spark = SparkSession.builder \
        .appName("TestIngestion") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()
    yield spark
    spark.stop()

def test_run_ingestion(spark_session):
    # Setup temporary directory for testing output
    output_dir = "data/test_outputs/ingested"
    input_path = "data/diabetes_sample_raw.csv"

    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)

    run_ingestion(input_path=input_path, output_path=output_dir)

    # Verify output file exists
    assert os.path.exists(output_dir)
    
    # Verify we can read the output CSV using Spark
    df = spark_session.read.csv(output_dir, header=True, inferSchema=True)
    assert df.count() > 0
    assert "race" in df.columns
    assert "gender" in df.columns
    assert "readmit_30_days" in df.columns
