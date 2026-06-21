# Limitations and Future Work

While our reweighing intervention successfully reduces demographic disparities in hospital readmission predictions, several limitations remain:

## 1. Demographic Representation & Group Sizes
* Certain demographic cohorts (e.g., Asian, Hispanic, and Unknown categories) have significantly smaller sample sizes compared to Caucasian and African American cohorts in the `diabetes_raw.csv` dataset.
* Smaller sample sizes increase variance in our inverse probability weights, making the model more sensitive to outliers within those underrepresented groups.

## 2. Proxy Variables & Feature Limitations
* This model relies on basic clinical proxies (e.g., `time_in_hospital` and `num_medications`). It does not incorporate social determinants of health (SDOH) such as income, transportation access, or primary care availability, which are strongly correlated with readmission risks and demographic groups.

## 3. Streaming and Live Drift
* Structured Streaming metrics are calculated per micro-batch. In a live production setting, if the distribution of incoming patient demographics shifts rapidly, the static offline weights computed during training may become stale, leading to **Model and Fairness Drift**.
* Future work should implement adaptive weighting systems or regular automated model re-training jobs triggered by live drift detection thresholds.
