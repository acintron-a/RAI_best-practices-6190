# ITCS 6190 – Cloud Computing for Data Analysis
## Course Project: Best-Practices in Algorithmic Fairness

### Project Goal
The goal of this ITCS 6190 course project is to design and implement a big data analytics pipeline using Apache Spark on a publicly available dataset. This project integrates Structured APIs, Streaming, and MLlib to demonstrate how these components work together to process, analyze, and model large-scale data. The workflow includes exploring a dataset, defining meaningful analytical and predictive questions, implementing scalable data processing and real-time streaming ingestion, applying machine learning, and communicating results effectively.

### Documentation Guide
For a comprehensive understanding of this project's background, approach, and outcomes, please refer to the detailed documentation located in the `docs/` directory.

We recommend reading the documentation in the following order:
1. **[Introduction](docs/introduction.md)**: An overview of Algorithmic Fairness, legal/regulatory frameworks, and the specific fairness context for this project.
2. **[Dataset Overview](docs/dataset_overview.md)**: Details on the dataset source, schema, distribution, and fairness considerations.
3. **[Methodology](docs/methodology.md)**: Explanation of the data transformations, Spark pipeline architecture, and bias mitigation strategies via Inverse Probability Weighting (IPW).
4. **[Results](docs/results.md)**: Outcomes of the experiments, including model performance metrics and fairness audits.
5. **[Limitations](docs/limitations.md)**: Caveats, assumptions, and potential shortcomings of the current approach.
6. **[Reproducibility Guide](docs/reproduction_guide.md)**: Step-by-step instructions to set up the environment and reproduce the batch and streaming pipelines locally.

*(A localized guide can also be found at `docs/README-documentation.md`)*

### Dataset Selection
This project utilizes the **Diabetes Hospital Readmission Dataset** (`fetch_diabetes_hospital`) available from the `fairlearn` Python module. It focuses on predicting hospital readmissions while ensuring demographic fairness across groups. 

*Note: In accordance with project guidelines, raw or full datasets are stored outside this GitHub repository. Only small representative samples (if any) are committed for testing.*

### Spark Components Implemented
This analytics pipeline incorporates all mandatory Apache Spark components:

* **Structured APIs**: The PySpark DataFrame API is extensively utilized throughout the pipeline (`src/ingestion.py`, `src/transformations.py`, and `src/ml_pipeline.py`) for robust data ingestion, cleaning, transformations, aggregations, and joins.
* **Spark SQL**: Spark SQL is used alongside the Structured APIs for analytical profiling and data transformation tasks.
* **Streaming**: A functional real-time streaming job is implemented using Spark Structured Streaming. The architecture simulates a live clinical stream via a TCP socket server (`src/patient_generator.py`) which reads records and broadcasts them. A real-time inference engine (`src/streaming.py`) connects to this source, applies the serialized MLlib PipelineModel, and outputs live metric updates (Selection Rates and False Negative Rates by Race) for every micro-batch.
* **MLlib**: Machine learning classification is performed using Spark MLlib (`src/ml_pipeline.py`). We evaluate an unmitigated `DecisionTreeClassifier` baseline, and then apply a sample reweighing bias mitigation strategy using IPW passed into the native `weightCol` parameter. Full feature transformations (VectorAssembler, StringIndexer) and model evaluation metrics are reported.
