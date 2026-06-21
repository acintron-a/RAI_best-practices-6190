import os
import subprocess
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from fairlearn.metrics import MetricFrame, false_negative_rate, selection_rate

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

def run_ml_pipeline(input_path="data/diabetes_transformed-csv", model_output_dir="outputs/models/fair_diabetes_model"):
    print("Initializing Spark session for MLlib Training Pipeline...")
    spark = SparkSession.builder \
        .appName("DiabetesMLPipeline") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()

    print(f"Loading transformed dataset from: {input_path}")
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Define feature indexing and assembly pipeline stages
    race_indexer = StringIndexer(inputCol="race", outputCol="race_index", handleInvalid="keep")
    gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_index", handleInvalid="keep")
    feature_cols = ["time_in_hospital", "num_medications", "race_index", "gender_index"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Split dataset into train and test sets (80% / 20%)
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    # Establish Baseline (Unmitigated) ML Model
    dt_baseline = DecisionTreeClassifier(labelCol="readmit_30_days", featuresCol="features", maxDepth=5, seed=42)
    baseline_pipeline = Pipeline(stages=[race_indexer, gender_indexer, assembler, dt_baseline])
    
    print("Training Baseline Model...")
    baseline_model = baseline_pipeline.fit(train_data)

    # Run predictions on the holdout test set
    baseline_preds_df = baseline_model.transform(test_data).select("race", "readmit_30_days", "prediction")
    baseline_preds = baseline_preds_df.toPandas()

    # Document Pre-Mitigation Disparities with Fairlearn
    metrics_before = MetricFrame(
        metrics={"FNR": false_negative_rate, "Selection_Rate": selection_rate},
        y_true=baseline_preds["readmit_30_days"],
        y_pred=baseline_preds["prediction"],
        sensitive_features=baseline_preds["race"]
    )
    print("\n=======================================================")
    print("[FAIRLEARN AUDIT] METRICS BEFORE IN-PROCESSING MITIGATION:")
    print("=======================================================")
    print(metrics_before.by_group)
    print(f"  Max FNR Disparity (difference): {metrics_before.difference()['FNR']:.4f}")
    print(f"  Max FNR Disparity (ratio):      {metrics_before.ratio()['FNR']:.4f}")

    # Calculate Sample Weights for In-Processing Mitigation (Inverse Probability Weighting)
    print("\nCalculating inverse-probability bias mitigation weights...")
    total_count = train_data.count()
    p_y = train_data.groupBy("readmit_30_days").count().withColumn("p_y", F.col("count") / total_count).drop("count")
    p_a = train_data.groupBy("race").count().withColumn("p_a", F.col("count") / total_count).drop("count")
    p_ya = train_data.groupBy("readmit_30_days", "race").count().withColumn("p_ya", F.col("count") / total_count).drop("count")

    weights_lookup = p_ya.join(p_y, on="readmit_30_days").join(p_a, on="race") \
        .withColumn("bias_mitigation_weight", (F.col("p_y") * F.col("p_a")) / F.col("p_ya"))

    # PERFORMANCE OPTIMIZATION: Broadcast join optimization. Since weights_lookup has very few rows (classes * sensitive groups),
    # broadcasting it avoids the expensive shuffle phase of a standard join on the training dataset.
    print("Applying Performance Optimization: Broadcasting weights lookup table.")
    train_data_weighted = train_data.join(
        F.broadcast(weights_lookup.select("readmit_30_days", "race", "bias_mitigation_weight")),
        on=["readmit_30_days", "race"],
        how="left"
    )

    # PERFORMANCE OPTIMIZATION: Cache optimization. We cache train_data_weighted as it will be loaded multiple times
    # during decision tree training and model evaluation.
    print("Applying Performance Optimization: Caching weighted training dataset.")
    train_data_weighted = train_data_weighted.cache()

    # Train the Mitigated ML Model using the native weightCol attribute
    dt_mitigated = DecisionTreeClassifier(
        labelCol="readmit_30_days",
        featuresCol="features",
        weightCol="bias_mitigation_weight",
        maxDepth=5,
        seed=42
    )

    mitigated_pipeline = Pipeline(stages=[race_indexer, gender_indexer, assembler, dt_mitigated])
    
    print("Training Mitigated Model...")
    final_fair_model = mitigated_pipeline.fit(train_data_weighted)

    # Run Predictions on the holdout test set
    mitigated_preds_df = final_fair_model.transform(test_data).select("race", "readmit_30_days", "prediction")
    mitigated_preds = mitigated_preds_df.toPandas()

    # Document Post-Mitigation Disparities with Fairlearn
    metrics_after = MetricFrame(
        metrics={"FNR": false_negative_rate, "Selection_Rate": selection_rate},
        y_true=mitigated_preds["readmit_30_days"],
        y_pred=mitigated_preds["prediction"],
        sensitive_features=mitigated_preds["race"]
    )
    print("\n=======================================================")
    print("[FAIRLEARN AUDIT] METRICS AFTER IN-PROCESSING MITIGATION:")
    print("=======================================================")
    print(metrics_after.by_group)
    print(f"  Max FNR Disparity (difference): {metrics_after.difference()['FNR']:.4f}")
    print(f"  Max FNR Disparity (ratio):      {metrics_after.ratio()['FNR']:.4f}")

    # Serialize and Save the pipeline model to disk
    print(f"\nPersisting bias-mitigated PipelineModel to directory: {model_output_dir}")
    os.makedirs(os.path.dirname(model_output_dir), exist_ok=True)
    final_fair_model.write().overwrite().save(model_output_dir)
    print("Model serialization complete!")

if __name__ == "__main__":
    run_ml_pipeline()
