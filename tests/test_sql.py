import pytest
from pyspark.sql import SparkSession

@pytest.fixture(scope="module")
def spark_session():
    spark = SparkSession.builder \
        .appName("TestSQL") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()
    yield spark
    spark.stop()

def test_spark_sql_queries(spark_session):
    # Create simple mock DataFrame
    data = [
        ("Caucasian", "Female", 1, 10, 0),
        ("AfricanAmerican", "Female", 3, 12, 1),
        ("Caucasian", "Male", 2, 8, 1),
        ("Asian", "Male", 1, 15, 0)
    ]
    columns = ["race", "gender", "time_in_hospital", "num_medications", "readmit_30_days"]
    df = spark_session.createDataFrame(data, columns)
    df.createOrReplaceTempView("test_diabetes_view")

    # Run SQL aggregation
    result = spark_session.sql("""
        SELECT race, AVG(time_in_hospital) as avg_time, SUM(readmit_30_days) as readmitted
        FROM test_diabetes_view
        GROUP BY race
    """).collect()

    # Verify query outputs
    races_collected = {row["race"]: (row["avg_time"], row["readmitted"]) for row in result}
    assert "Caucasian" in races_collected
    assert races_collected["Caucasian"] == (1.5, 1)
    assert races_collected["AfricanAmerican"] == (3.0, 1)
