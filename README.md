# ITCS 6190 – Cloud Computing for Data Analysis
## Course Project: Best-Practices in Algorithmic Fairness

This project is motivated by a subfield of artificial intelligence and machine learning called “Algorithmic Fairness”, which is dedicated to ensure that automated decision-making systems do not discriminate against individuals or groups based on protected or sensitive traits (e.g., race, gender, age, socioeconomic status, etc.).

Sensitive features are determined by human beings while addressing: “Which groups of individuals are at risk of experiencing harm from a machine learning model?”

---

## 1. Algorithmic Fairness Framework & Context

There are three main criteria in the determination process:

### 1.1 Legal and Regulatory Frameworks (Protected Attributes)
In many traditional use cases (such as lending, housing, and hiring), sensitive features are dictated by anti-discrimination laws.
* Under frameworks like the US Civil Rights Act or the EU General Data Protection Regulation (GDPR), certain characteristics are designated as **"protected classes."**
* This is why datasets like `fetch_credit_card()` inherently label variables like **Sex, Race, and Age** as sensitive features. The law explicitly forbids using these attributes to deny people baseline opportunities.

### 1.2 The Type of Algorithmic Harm Being Assessed
The type of harm dictates what feature becomes "sensitive":
* **Allocation Harms:** Occur when an AI system unfairly extends or withholds opportunities, resources, or information.
  * *Example:* In `fetch_credit_card()`, **Age** and **Marriage Status** are designated as sensitive because a bank's model might unfairly withhold credit lines from younger or single applicants.
* **Quality of Service Harms:** Occur when an AI system simply works much better for one group of people than another, even if no money or resources are explicitly on the line.
  * *Example:* In a speech-to-text algorithm or facial recognition system, **Native Language Status** or skin tone (**Melanin Content**) would be determined as the sensitive features, because the technical performance of the model risks degrading for those specific sub-populations.

### 1.3 Contextual and Domain-Specific Vetting
Sometimes, a sensitive feature has nothing to do with demographics. It is determined purely by the domain context where the model is being deployed.
* In a healthcare setting (like Fairlearn's `fetch_diabetes_hospital()` dataset), features like **Insurance Type (Medicare/Medicaid)** or **Race** are designated as sensitive features.
* While insurance status is not an immutable demographic trait, an algorithm that unintentionally prioritizes commercially insured patients over Medicaid patients for hospital re-admission care would create a massive systemic inequality.

| Machine Learning Type | Fairness Instance | Case Details / Notes | Reference |
| :--- | :--- | :--- | :--- |
| **Regression** | Sickness Severity Score | <ul><li>To determine if individual may require care in future, a risk score was produced.</li><li>Observations showed that Black enrollees have higher rate of chronic illnesses than White enrollees of same score.</li><li>Algorithm used health costs as proxy for health.</li></ul> | Obermeyer, Ziad, Brian Powers, Christine Vogeli, and Sendhil Mullainathan. "Dissecting racial bias in an algorithm used to manage the health of populations." Science 366, no. 6464 (2019). / doi:10.1126/science.aax2342. |
| **Classification** | Credit Worthiness | <ul><li>To determine if applicant may default, a probability score was used.</li><li>“In default”: failed to pay a debt for longer than 90 days in more than one account over 18-24 month period.</li><li>Use equalized odds to enforce accuracy is equal in all groups.</li></ul> | Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. arXiv. https://doi.org/10.48550/arXiv.1610.02413 |
| **Ranking** | Job Ranking | <ul><li>Ranked lists produced by biased model can amplify bias.</li><li>Without fairness intervention, top ranked results can be skewed with respect to sensitive attributes.</li><li>To mitigate, fair re-ranking algorithms are required (e.g., score maximizing greedy mitigation).</li></ul> | Geyik, Sahin Cem, Stuart Ambler, and Krishnaram Kenthapadi. "Fairness-Aware Ranking in Search & Recommendation Systems with Application to LinkedIn Talent Search." Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD) (2019). / arXiv:1905.01989. |
| **Recommendation** | Targeted Advertising | <ul><li>Recommendations produced by biased models may amplify bias.</li><li>Experiment: Different ad settings led to different frequency of high paying jobs shown.</li><li>Require multi-sided fairness or sequential recommendations.</li></ul> | Pitoura, E., Stefanidis, K., & Koutrika, G. (2021). Fairness in rankings and recommendations: An overview. arXiv. https://doi.org/10.48550/arXiv.2104.05994 |
| **Clustering** | Fair Hiring | <ul><li>Cluster resumes for shortlist in a hiring scenario.</li><li>Callback rates may differ per group (incorporate fairness constraints in optimization).</li><li>Aim for proportional representation of the sensitive class per cluster.</li></ul> | Abraham, Savitha Sam, Deepak P., and Sanil V. "Fairness in Clustering with Multiple Sensitive Attributes." Proceedings of the 23rd International Conference on Extending Database Technology (EDBT) (2020). / arXiv:1910.05113. |
| **Anomaly Detection** | Government Relief Fund | <ul><li>Small fraction of COVID relief direct payments experience fraud.</li><li>Scale of money allocation makes traditional tracking impossible.</li><li>Outliers should not be skewed towards a particular group.</li></ul> | Deepak, P., and Savitha Sam Abraham. "FairLOF: Fairness in Outlier Detection." Applied Intelligence 51, no. 12 (2021). / doi:10.1007/s41019-021-00169-x. |

---

## 2. Technical Implementation Narrative Plan

To analyze this problem at scale, this project uses the `fetch_diabetes_hospital()` dataset from Fairlearn (`101,766` clinical observations) to predict 30-day patient readmissions (`readmit_30_days`). 

Due to the size of this dataset, loading and writing data partitions locally can be slow. This pipeline uses **Apache Spark** to distribute feature transformations and run real-time streaming operations across three core milestones.

### Milestone 1: Data Ingestion + EDA (Batch/Offline Layer)
* **Objective:** Ingest the clinical dataset using Spark’s Structured DataFrame API, analyze demographic representations, perform baseline model training, and compute sample-reweighing mitigation factors.
* **Approach:** Parse and handle the raw clinical dataset using explicit schemas to prevent data corruption during type casting.
  * Execute distributed `.groupBy()` aggregations to calculate historical baseline re-admission rates across sensitive demographic cohorts (`race` and `gender`).
  * Train an unweighted baseline `pyspark.ml.classification.DecisionTreeClassifier` and output test predictions into a local `fairlearn.metrics.MetricFrame` to document "Pre-Mitigation" discrepancies in False Negative Rates (FNR) and Selection Rates.
  * Use Spark DataFrame transformations to calculate inverse-probability reweighing variables that balance representation across groups, appending a computed `bias_mitigation_weight` column back into the distributed training matrix.

### Milestone 2: Streaming + MLlib (Real-Time Inference Layer)
* **Objective:** Train and serialize the final fair model, build a live network stream simulator, and implement a low-latency Spark Structured Streaming query that applies the model to incoming data.
* **Approach:**
  * Train a mitigated `DecisionTreeClassifier` by passing the calculated `bias_mitigation_weight` column directly into the tree's native `weightCol` parameter. Verify the reduction in disparity using a post-mitigation Fairlearn audit, and save the final pipeline as a complete `PipelineModel` directory on disk.
  * Build a standalone socket stream generator script (`patient_generator.py`) that acts as a mock server, emitting real-time JSON strings representing incoming patient profiles (demographics, medication counts, and hospital stay duration).
  * Build a streaming inference engine (`project_stream.py`) that establishes a socket connection via `spark.readStream`, parses incoming patient data fields against a fixed schema, and loads the pre-trained `PipelineModel` to score readmission risks on the fly.
  * Implement streaming group transformations that aggregate predictions by demographics in real time, allowing operators to monitor the system's live selection rates and spot potential model drift.

### Milestone 3: Full Pipeline + Documentation (Integration & Quality Assurance)
* **Objective:** Create a single-command execution script to run the entire pipeline, structure the project repository cleanly, and write a thorough final report that analyzes both model performance and fairness trade-offs.
* **Approach:**
  * Implement an automated automation shell script (`run.sh`) that coordinates the workflow: runs the offline training script, spins up the background streaming socket server, and initializes the real-time PySpark inference query.
  * Manage repository size constraints by tracking a small, 5-row sample file within the Git directory for local validation, while updating the primary scripts to point to large external storage files.
  * Use terminal log traces and screenshots to document successful streaming queries, and update the final project report to compare overall accuracy metrics against the reduction in demographic group disparities.

---

## 3. Project Reproducibility Workflow

To run the complete end-to-end pipeline locally, ensure `pyspark` and `fairlearn` are installed, and execute the following scripts in order across separate terminal sessions:

1. **Step 1: Train the Fair Model (Offline Layer)**
   ```bash
   python project_train.py
   ```
2. **Step 2: Start the Live Patient Feed**
```bash
python patient_generator.py
```
3. ***Step 3: Execute Streaming Inference and Live Monitoring***
```bash
spark-submit project_stream.py
```  
  
