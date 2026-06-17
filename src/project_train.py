import os
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from fairlearn.metrics import MetricFrame, false_negative_rate, selection_rate

# Initialize local Spark session optimized for handling data structures
spark = SparkSession.builder \
    .appName("DiabetesFairnessOfflineTraining") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

print("--- STAGE 1 & 2: INGESTION AND DISPARITY AUDITING ---")
# Ingest data using Structured DataFrame API (Fulfills Component 1 of Course Project)
# (Replacing with your approved hospital dataset pathway)
df = spark.read.csv("data/diabetes_raw.csv", header=True, inferSchema=True)

# Define pipeline stages for clinical and demographic feature engineering
race_indexer = StringIndexer(inputCol="race", outputCol="race_index", handleInvalid="keep")
gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_index", handleInvalid="keep")
feature_cols = ["time_in_hospital", "num_medications", "race_index", "gender_index"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Split data into train and test sets to mimic production deployment steps
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Establish Baseline (Unmitigated) Pipeline Model
dt_baseline = DecisionTreeClassifier(labelCol="readmit_30_days", featuresCol="features", maxDepth=5, seed=42)
baseline_pipeline = Pipeline(stages=[race_indexer, gender_indexer, assembler, dt_baseline])
baseline_model = baseline_pipeline.fit(train_data)

# Run Pre-Mitigation Baseline Predictions for Fairlearn Audit
baseline_preds = baseline_model.transform(test_data).select("race", "readmit_30_days", "prediction").toPandas()

metrics_before = MetricFrame(
    metrics={"FNR": false_negative_rate, "Selection_Rate": selection_rate},
    y_true=baseline_preds["readmit_30_days"],
    y_pred=baseline_preds["prediction"],
    sensitive_features=baseline_preds["race"]
)
print("\n[FAIRLEARN AUDIT] METRICS BEFORE IN-PROCESSING MITIGATION:")
print(metrics_before.by_group)
print(f"  Max FNR Disparity (difference): {metrics_before.difference()}")
print(f"  Max FNR Disparity (ratio):      {metrics_before.ratio()}")

print("\n--- STAGE 3: IN-PROCESSING MITIGATION VIA BIAS REWEIGHING ---")
# Calculate sample weights directly via distributed aggregations to eliminate demographic correlation
total_count = train_data.count()
p_y = train_data.groupBy("readmit_30_days").count().withColumn("p_y", F.col("count") / total_count).drop("count")
p_a = train_data.groupBy("race").count().withColumn("p_a", F.col("count") / total_count).drop("count")
p_ya = train_data.groupBy("readmit_30_days", "race").count().withColumn("p_ya", F.col("count") / total_count).drop("count")

weights_lookup = p_ya.join(p_y, on="readmit_30_days").join(p_a, on="race") \
    .withColumn("bias_mitigation_weight", (F.col("p_y") * F.col("p_a")) / F.col("p_ya"))

# Join weights column back natively into our training matrix
train_data_weighted = train_data.join(
    weights_lookup.select("readmit_30_days", "race", "bias_mitigation_weight"), 
    on=["readmit_30_days", "race"], 
    how="left"
)

# Initialize Classifier using the native weightCol attribute
dt_mitigated = DecisionTreeClassifier(
    labelCol="readmit_30_days", 
    featuresCol="features", 
    weightCol="bias_mitigation_weight", 
    maxDepth=5, 
    seed=42
)

# Re-train unified pipeline incorporating mathematical bias mitigation
mitigated_pipeline = Pipeline(stages=[race_indexer, gender_indexer, assembler, dt_mitigated])
final_fair_model = mitigated_pipeline.fit(train_data_weighted)

# Run Post-Mitigation Predictions for Fairlearn Validation
mitigated_preds = final_fair_model.transform(test_data).select("race", "readmit_30_days", "prediction").toPandas()

metrics_after = MetricFrame(
    metrics={"FNR": false_negative_rate, "Selection_Rate": selection_rate},
    y_true=mitigated_preds["readmit_30_days"],
    y_pred=mitigated_preds["prediction"],
    sensitive_features=mitigated_preds["race"]
)
print("\n[FAIRLEARN AUDIT] METRICS AFTER IN-PROCESSING MITIGATION:")
print(metrics_after.by_group)
print(f"  Max FNR Disparity (difference): {metrics_after.difference()}")
print(f"  Max FNR Disparity (ratio):      {metrics_after.ratio()}")

print("\n--- STAGE 4: PERSISTING PIPELINE MODEL SANAPSHOT ---")
# Export the entire pipeline model (including indexers) to disk (Matches HO-L8 Task 4 & 5 design)
MODEL_DIR = "outputs/models/fair_diabetes_model"
final_fair_model.write().overwrite().save(MODEL_DIR)
print(f"Mitigated PipelineModel successfully serialized to folder: '{MODEL_DIR}'")