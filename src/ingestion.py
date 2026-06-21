import os
import subprocess
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType

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

def run_ingestion(input_path="data/diabetes_raw.csv", output_path="data/diabetes_ingested-csv"):
    print("Initializing Spark session for Data Ingestion...")
    spark = SparkSession.builder \
        .appName("DiabetesDataIngestion") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()

    # Define explicit schema for target features to prevent type-casting errors
    schema = StructType([
        StructField("race", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("age", StringType(), True),
        StructField("discharge_disposition_id", StringType(), True),
        StructField("admission_source_id", StringType(), True),
        StructField("time_in_hospital", IntegerType(), True),
        StructField("medical_specialty", StringType(), True),
        StructField("num_lab_procedures", IntegerType(), True),
        StructField("num_procedures", IntegerType(), True),
        StructField("num_medications", IntegerType(), True),
        StructField("primary_diagnosis", StringType(), True),
        StructField("number_diagnoses", IntegerType(), True),
        StructField("max_glu_serum", StringType(), True),
        StructField("A1Cresult", StringType(), True),
        StructField("insulin", StringType(), True),
        StructField("change", StringType(), True),
        StructField("diabetesMed", StringType(), True),
        StructField("medicare", BooleanType(), True),
        StructField("medicaid", BooleanType(), True),
        StructField("had_emergency", BooleanType(), True),
        StructField("had_inpatient_days", BooleanType(), True),
        StructField("had_outpatient_days", BooleanType(), True),
        StructField("readmitted", StringType(), True),
        StructField("readmit_binary", IntegerType(), True),
        StructField("readmit_30_days", IntegerType(), True)
    ])

    print(f"Reading raw CSV data from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input data not found at: {input_path}")

    df = spark.read.csv(input_path, header=True, schema=schema)
    
    print("\n--- SCHEMA ANALYSIS ---")
    df.printSchema()

    # Register as Temp View for Spark SQL
    df.createOrReplaceTempView("diabetes_raw_view")

    print("\n--- EXPLORATORY DATA ANALYSIS (SPARK SQL) ---")
    # Query 1: Total records and distribution of readmissions
    print("\n[EDA Query 1] Overall readmission distribution (readmit_30_days):")
    spark.sql("""
        SELECT readmit_30_days, COUNT(*) as count, ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
        FROM diabetes_raw_view
        GROUP BY readmit_30_days
    """).show()

    # Query 2: Historical baseline readmission rate by Race
    print("\n[EDA Query 2] Baseline readmission rates by Race (Sensitive Attribute 1):")
    spark.sql("""
        SELECT race, 
               COUNT(*) as total_patients,
               SUM(readmit_30_days) as readmitted_count,
               ROUND(SUM(readmit_30_days) * 1.0 / COUNT(*), 4) as readmission_rate
        FROM diabetes_raw_view
        GROUP BY race
        ORDER BY readmission_rate DESC
    """).show()

    # Query 3: Historical baseline readmission rate by Gender
    print("\n[EDA Query 3] Baseline readmission rates by Gender (Sensitive Attribute 2):")
    spark.sql("""
        SELECT gender, 
               COUNT(*) as total_patients,
               SUM(readmit_30_days) as readmitted_count,
               ROUND(SUM(readmit_30_days) * 1.0 / COUNT(*), 4) as readmission_rate
        FROM diabetes_raw_view
        GROUP BY gender
        ORDER BY readmission_rate DESC
    """).show()

    # Query 4: Cross-tabulation of Race and Gender baseline rates
    print("\n[EDA Query 4] Joint distribution of Race and Gender readmission rates:")
    spark.sql("""
        SELECT race, gender,
               COUNT(*) as count,
               ROUND(AVG(readmit_30_days), 4) as readmission_rate
        FROM diabetes_raw_view
        GROUP BY race, gender
        ORDER BY race, gender
    """).show()

    print(f"Persisting ingested DataFrame to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.write.mode("overwrite").option("header", "true").csv(output_path)
    print("Ingestion phase complete!")

if __name__ == "__main__":
    run_ingestion()
