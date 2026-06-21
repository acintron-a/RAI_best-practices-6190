# Introduction 

This project is motivated by "Algorithmic Fairness," a subfield of artificial intelligence and machine learning dedicated to ensuring that automated decision-making systems do not discriminate against individuals based on protected or sensitive traits (e.g., race, gender, age, or socioeconomic status).

To explore these concepts, we utilize the `fetch_diabetes_hospital` dataset from the `fairlearn` Python module (more details about this data are available in `dataset_overview.md`). In healthcare settings, hospital readmission prediction models are frequently used to allocate limited resources, such as enrolling high-risk patients in specialized post-discharge care programs. 

If a prediction model suffers from demographic bias—such as underpredicting readmission risks for specific groups—it can lead to **Allocation Harm**, where critical care is unfairly withheld from historically underserved populations.

To prevent this, sensitive features are determined by human practitioners by asking: "Which groups of individuals are at risk of experiencing harm from this machine learning model?"

There are three main criteria in the determination process:

### 1. Legal and Regulatory Frameworks (Protected Attributes)
In many traditional use cases (such as lending, housing, and hiring), sensitive features are dictated by anti-discrimination laws.

* Under frameworks like the US Civil Rights Act or the EU General Data Protection Regulation (GDPR), certain characteristics are designated as **"protected classes."**
* This is why datasets like `fetch_diabetes_hospital()` inherently label variables like **Race, Gender, and Age** as sensitive features. The law explicitly forbids using these attributes to deny people baseline opportunities or care.
### 2. The Type of Algorithmic Harm Being Assessed
The type of harm dictates what feature becomes "sensitive":

* **Allocation Harms:** Occur when an AI system unfairly extends or withholds opportunities, resources, or information.
* *Example:* In `fetch_diabetes_hospital()`, **Race** and **Gender** are designated as sensitive because a healthcare model might unfairly withhold post-discharge care programs from specific demographic groups.

* **Quality of Service Harms:** Occur when an AI system simply works much better for one group of people than another, even if no money or resources are explicitly on the line.
* *Example:* In a speech-to-text algorithm or facial recognition system, **Native Language Status** or skin tone (**Melanin Content**) would be determined as the sensitive features, because the technical performance of the model risks degrading for those specific sub-populations.

### 3. Contextual and Domain-Specific Vetting

Sometimes, a sensitive feature has nothing to do with demographics. It is determined purely by the domain context where the model is being deployed.

* In a healthcare setting (like Fairlearn's `fetch_diabetes_hospital()` dataset), features like **Insurance Type (Medicare/Medicaid)** are designated as sensitive features.
* While having Medicaid is not an immutable demographic trait, an algorithm that unintentionally prioritizes commercially insured patients over Medicaid patients for hospital re-admission care would create a massive systemic inequality.


## Machine Learning Fairness Contexts (Table)
In the table below we summarize machine learning model types and some fairness context.

| Machine Learning Type | Fairness Instance | Case Details / Notes | Reference |
| :--- | :--- | :--- | :--- |
| **Regression** | Sickness Severity Score | <ul><li>To determine if individual may require care in future, a risk score was produced.</li><li>Observations showed that Black enrollees have higher rate of chronic illnesses than White enrollees of same score.</li><li>Algorithm used health costs as proxy for health.</li></ul> | Obermeyer, Ziad, Brian Powers, Christine Vogeli, and Sendhil Mullainathan. "Dissecting racial bias in an algorithm used to manage the health of populations." Science 366, no. 6464 (2019). / doi:10.1126/science.aax2342. |
| **Classification** | Credit Worthiness | <ul><li>To determine if applicant may default, a probability score was used.</li><li>“In default”: failed to pay a debt for longer than 90 days in more than one account over 18-24 month period.</li><li>Use equalized odds to enforce accuracy is equal in all groups.</li></ul> | Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. arXiv. https://doi.org/10.48550/arXiv.1610.02413 |
| **Ranking** | Job Ranking | <ul><li>Ranked lists produced by biased model can amplify bias.</li><li>Without fairness intervention, top ranked results can be skewed with respect to sensitive attributes.</li><li>To mitigate, fair re-ranking algorithms are required (e.g., score maximizing greedy mitigation).</li></ul> | Geyik, Sahin Cem, Stuart Ambler, and Krishnaram Kenthapadi. "Fairness-Aware Ranking in Search & Recommendation Systems with Application to LinkedIn Talent Search." Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD) (2019). / arXiv:1905.01989. |
| **Recommendation** | Targeted Advertising | <ul><li>Recommendations produced by biased models may amplify bias.</li><li>Experiment: Different ad settings led to different frequency of high paying jobs shown.</li><li>Require multi-sided fairness or sequential recommendations.</li></ul> | Pitoura, E., Stefanidis, K., & Koutrika, G. (2021). Fairness in rankings and recommendations: An overview. arXiv. https://doi.org/10.48550/arXiv.2104.05994 |
| **Clustering** | Fair Hiring | <ul><li>Cluster resumes for shortlist in a hiring scenario.</li><li>Callback rates may differ per group (incorporate fairness constraints in optimization).</li><li>Aim for proportional representation of the sensitive class per cluster.</li></ul> | Abraham, Savitha Sam, Deepak P., and Sanil V. "Fairness in Clustering with Multiple Sensitive Attributes." Proceedings of the 23rd International Conference on Extending Database Technology (EDBT) (2020). / arXiv:1910.05113. |
| **Anomaly Detection** | Government Relief Fund | <ul><li>Small fraction of COVID relief direct payments experience fraud.</li><li>Scale of money allocation makes traditional tracking impossible.</li><li>Outliers should not be skewed towards a particular group.</li></ul> | Deepak, P., and Savitha Sam Abraham. "FairLOF: Fairness in Outlier Detection." Applied Intelligence 51, no. 12 (2021). / doi:10.1007/s41019-021-00169-x. |

[Return to Documentation Guide](./README-documentation.md)
