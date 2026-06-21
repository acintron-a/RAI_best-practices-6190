import os
import subprocess
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
from pyspark.ml import PipelineModel

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

def run_streaming(host="localhost", port=9999, model_dir="outputs/models/fair_diabetes_model", output_logs="outputs/project_live_fairness_logs"):
    print("Initializing low-latency Structured Streaming session...")
    spark = SparkSession.builder \
        .appName("DiabetesRealTimeInferenceAndFairnessMonitor") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()

    # Define the schema matching patient_generator.py output
    patient_schema = StructType([
        StructField("race", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("time_in_hospital", IntegerType(), True),
        StructField("num_medications", IntegerType(), True),
        StructField("readmit_30_days", IntegerType(), True),
        StructField("timestamp", StringType(), True)
    ])

    print(f"Connecting to Structured Streaming socket source at {host}:{port}...")
    raw_stream = spark.readStream \
        .format("socket") \
        .option("host", host) \
        .option("port", port) \
        .load()

    # Parse incoming raw JSON string using the schema
    parsed_patients = raw_stream.selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), patient_schema).alias("data")) \
        .select("data.*")

    # Cast timestamp
    parsed_patients = parsed_patients.withColumn("timestamp", col("timestamp").cast(TimestampType()))

    print(f"Loading pre-trained bias-mitigated PipelineModel from: {model_dir}")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Trained model not found at: {model_dir}. Please run the training script first.")

    fair_pipeline_model = PipelineModel.load(model_dir)

    # Apply the pipeline model to the streaming DataFrame
    predictions_stream = fair_pipeline_model.transform(parsed_patients)

    # Custom micro-batch processing logic to evaluate metrics and dump logs
    def process_micro_batch(batch_df, batch_id):
        print(f"\n=======================================================")
        print(f" LIVE SYSTEM FAIRNESS MONITOR AUDIT - BATCH WINDOW: {batch_id}")
        print(f"=======================================================")

        if batch_df.isEmpty():
            print("  (empty micro-batch — awaiting incoming patient stream data)")
            return

        # 1. Print Selection Rate by Race
        print("\n[SELECTION RATES] Real-time decision distribution by demographic group:")
        batch_df.groupBy("race", "prediction").count() \
            .orderBy("race", "prediction").show(truncate=False)

        # 2. Print Live False Negative Rate (FNR) by Race
        actual_positives = batch_df.filter(col("readmit_30_days") == 1)
        ap_count = actual_positives.count()

        if ap_count > 0:
            fnr_by_race = actual_positives.groupBy("race").agg(
                F.count("*").alias("actual_positives"),
                F.sum(F.when(col("prediction") == 0.0, 1).otherwise(0)).alias("false_negatives")
            ).withColumn("FNR", F.round(col("false_negatives") / col("actual_positives"), 4))

            print("[FALSE NEGATIVE RATES] Real-time FNR by demographic group:")
            fnr_by_race.orderBy("race").show(truncate=False)
        else:
            print("  No actual positive labels in this micro-batch; FNR cannot be computed.")

        # 3. Append predictions to historical logs directory
        print(f"Appending micro-batch outcomes to: {output_logs}")
        batch_df.select("race", "gender", "time_in_hospital", "num_medications", "readmit_30_days", "prediction").coalesce(1).write \
            .mode("append") \
            .option("header", "true") \
            .csv(output_logs)

    # Execute stream with foreachBatch
    query = predictions_stream.writeStream \
        .outputMode("append") \
        .foreachBatch(process_micro_batch) \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    run_streaming()