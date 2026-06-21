import os
import subprocess
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# -------------------------------------------------------------------------
# ENVIRONMENT SETUP (Ensuring JVM compatibility as in templates)
# -------------------------------------------------------------------------
if os.path.exists("/usr/libexec/java_home"):
    try:
        os.environ["JAVA_HOME"] = subprocess.check_output(["/usr/libexec/java_home", "-v", "17"]).decode("utf-8").strip()
    except Exception:
        pass

os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--driver-java-options \""
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
    "--add-opens=java.base/java.io=ALL-UNNAMED "
    "--add-opens=java.base/java.net=ALL-UNNAMED "
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
    "--add-opens=java.base/java.util.logging=ALL-UNNAMED "
    "--add-opens=java.security/java.security=ALL-UNNAMED "
    "--add-opens=java.base/sun.misc=ALL-UNNAMED "
    "\" pyspark-shell"
)

def run_transformations(input_path="data/diabetes_ingested-csv", output_path="data/diabetes_transformed-csv"):
    print("Initializing Spark session for Data Transformations...")
    spark = SparkSession.builder \
        .appName("DiabetesDataTransformations") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()

    print(f"Reading ingested data from: {input_path}")
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Clean the demographic columns (handle potential nulls or Unknown values)
    # We replace nulls or empty strings with "Unknown" for race/gender
    df_cleaned = df.withColumn("race", when(col("race").isNull() | (col("race") == ""), "Unknown").otherwise(col("race"))) \
                   .withColumn("gender", when(col("gender").isNull() | (col("gender") == ""), "Unknown").otherwise(col("gender")))

    # Clean target columns: convert readmit_30_days, time_in_hospital, and num_medications to correct integer types
    df_cleaned = df_cleaned.withColumn("time_in_hospital", col("time_in_hospital").cast("integer")) \
                           .withColumn("num_medications", col("num_medications").cast("integer")) \
                           .withColumn("readmit_30_days", col("readmit_30_days").cast("integer"))

    # Register as Temp View for Spark SQL transformations and analytical queries
    df_cleaned.createOrReplaceTempView("diabetes_cleaned_view")

    print("\n--- CLINICAL METRIC PROFILING (SPARK SQL) ---")
    
    # Query 1: Average time in hospital and medications by demographic groups
    print("\n[SQL Query 1] Average time in hospital & medications by Race and Gender:")
    spark.sql("""
        SELECT race, gender,
               ROUND(AVG(time_in_hospital), 2) as avg_time_in_hospital,
               ROUND(AVG(num_medications), 2) as avg_num_medications,
               COUNT(*) as cohort_size
        FROM diabetes_cleaned_view
        GROUP BY race, gender
        ORDER BY race, gender
    """).show()

    # Query 2: Readmissions profile based on clinical indicators
    print("\n[SQL Query 2] Average time in hospital and number of medications for readmitted vs non-readmitted patients:")
    spark.sql("""
        SELECT readmit_30_days,
               ROUND(AVG(time_in_hospital), 2) as avg_time_in_hospital,
               ROUND(AVG(num_medications), 2) as avg_num_medications,
               COUNT(*) as patient_count
        FROM diabetes_cleaned_view
        GROUP BY readmit_30_days
        ORDER BY readmit_30_days
    """).show()

    # Select only the features we need for our ML Pipeline model to keep it clean and minimal
    target_columns = ["race", "gender", "time_in_hospital", "num_medications", "readmit_30_days"]
    df_transformed = df_cleaned.select(*target_columns)

    print(f"Persisting cleaned and transformed DataFrame to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_transformed.write.mode("overwrite").option("header", "true").csv(output_path)
    print("Transformation phase complete!")

if __name__ == "__main__":
    run_transformations()
