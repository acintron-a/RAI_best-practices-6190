
import os
import subprocess
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
from pyspark.ml import PipelineModel

# -------------------------------------------------------------------------
# ENVIRONMENT SETUP (From Lab HO-L8 templates to ensure JVM compatibility)
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
    "--add-opens=java.base/java.security=ALL-UNNAMED "
    "--add-opens=java.base/sun.misc=ALL-UNNAMED "
    "\" pyspark-shell"
)

# -------------------------------------------------------------------------
# CORE STREAMING APPLICATION
# -------------------------------------------------------------------------

# Initialize low-latency streaming Spark engine
spark = SparkSession.builder \
    .appName("DiabetesRealTimeInferenceAndFairnessMonitor") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

# 1. Define the Schema matching the exact output of patient_generator.py
patient_schema = StructType([
    StructField("race", StringType(), True),
    StructField("gender", StringType(), True),
    StructField("time_in_hospital", IntegerType(), True),  # Matches the int() cast in the generator
    StructField("num_medications", IntegerType(), True),   # Matches the int() cast in the generator
    StructField("readmit_30_days", IntegerType(), True),   # Ground truth label for live FNR computation
    StructField("timestamp", StringType(), True)
])

print("--- STAGE 5: INITIALIZING STRUCTURED STREAMING SOCKET SOURCE ---")
# Establish connection to live patient network generator
raw_stream = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# Parse the incoming raw JSON stream using the explicit schema
parsed_patients = raw_stream.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), patient_schema).alias("data")) \
    .select("data.*")

# Convert timestamp column to proper timestamp formatting
parsed_patients = parsed_patients.withColumn("timestamp", col("timestamp").cast(TimestampType()))

print("--- STAGE 6: APPLICATION OF LIVE ML PIPELINE MODEL ---")
# Load the pre-trained, bias-mitigated PipelineModel saved by project_train.py
MODEL_DIR = "outputs/models/fair_diabetes_model"
try:
    fair_pipeline_model = PipelineModel.load(MODEL_DIR)
except Exception as e:
    print(f"\nCRITICAL ERROR: Could not load model from '{MODEL_DIR}'.")
    print("Did you run the batch training script (`project_train.py`) first to generate the model?")
    raise e

# Apply the ML model directly onto the incoming data stream
predictions_stream = fair_pipeline_model.transform(parsed_patients)

print("--- STAGE 7: STREAMING AGGREGATION & REAL-TIME DRIFT MONITORING ---")

# Custom callback function to evaluate streaming fairness metrics per micro-batch
def process_micro_batch(batch_df, batch_id):
    print(f"\n=======================================================")
    print(f" LIVE SYSTEM FAIRNESS MONITOR AUDIT - BATCH WINDOW: {batch_id}")
    print(f"=======================================================")

    if batch_df.isEmpty():
        print("  (empty batch — awaiting patient data)")
        return

    # --- Selection Rate Distribution ---
    print("\n[SELECTION RATES] Prediction distribution by demographic group:")
    batch_df.groupBy("race", "prediction").count() \
        .orderBy("race", "prediction").show(truncate=False)

    # --- Live False Negative Rate (FNR) by Race ---
    # FNR = FalseNegatives / ActualPositives = count(actual=1, pred=0) / count(actual=1)
    actual_positives = batch_df.filter(col("readmit_30_days") == 1)
    ap_count = actual_positives.count()

    if ap_count > 0:
        fnr_by_race = actual_positives.groupBy("race").agg(
            F.count("*").alias("actual_positives"),
            F.sum(F.when(col("prediction") == 0.0, 1).otherwise(0)).alias("false_negatives")
        ).withColumn("FNR", F.round(col("false_negatives") / col("actual_positives"), 4))

        print("[FALSE NEGATIVE RATES] Live FNR by demographic group:")
        fnr_by_race.orderBy("race").show(truncate=False)
    else:
        print("  No actual positives in this batch — FNR not computable.")

    # Save raw predictions to disk for historical model drift tracking
    batch_df.select("race", "readmit_30_days", "prediction").coalesce(1).write \
        .mode("append") \
        .option("header", "true") \
        .csv("outputs/project_live_fairness_logs")

# Execute streaming pipeline using foreachBatch on raw predictions for full metric access
query = predictions_stream.writeStream \
    .outputMode("append") \
    .foreachBatch(process_micro_batch) \
    .start()

query.awaitTermination()