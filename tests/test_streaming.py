import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

@pytest.fixture(scope="module")
def spark_session():
    spark = SparkSession.builder \
        .appName("TestStreamingSchema") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()
    yield spark
    spark.stop()

def test_streaming_json_parsing(spark_session):
    # Match the schema defined in streaming.py
    patient_schema = StructType([
        StructField("race", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("time_in_hospital", IntegerType(), True),
        StructField("num_medications", IntegerType(), True),
        StructField("readmit_30_days", IntegerType(), True),
        StructField("timestamp", StringType(), True)
    ])

    # Mock dynamic socket message
    mock_data = [('{"race":"Caucasian","gender":"Female","time_in_hospital":3,"num_medications":18,"readmit_30_days":1,"timestamp":"2026-06-20 12:00:00"}',)]
    df = spark_session.createDataFrame(mock_data, ["value"])

    # Parse and extract
    parsed_df = df.select(from_json(col("value"), patient_schema).alias("data")).select("data.*")

    # Collect and assert
    row = parsed_df.collect()[0]
    assert row["race"] == "Caucasian"
    assert row["gender"] == "Female"
    assert row["time_in_hospital"] == 3
    assert row["num_medications"] == 18
    assert row["readmit_30_days"] == 1
    assert row["timestamp"] == "2026-06-20 12:00:00"
