# Dataset Overview - Diabetes Hospital Readmissions

This project utilizes the `fetch_diabetes_hospital()` dataset from the Fairlearn Python library, containing clinical records of diabetic patients from 130 US hospitals (1999–2008).

## Dataset Dimensions
* **Total Rows**: 101,766
* **Total Columns**: 25 (Raw), 24 Features, 1 Target

## Data Dictionary (Schema)

The dataset contains the following variables used in the bias-mitigation and readmission prediction pipeline:

| Variable Name | Data Type | Role | Description | Valid Values / Range |
| :--- | :--- | :--- | :--- | :--- |
| **race** | Categorical | Feature (Sensitive) | Race of the patient | African American, Asian, Caucasian, Hispanic, Other, Unknown |
| **gender** | Categorical | Feature (Sensitive) | Gender of the patient | Female, Male |
| **age** | Categorical | Feature (Sensitive) | Grouped age brackets | '30 years or younger', '30-60 years', 'Over 60 years' |
| **admission_source_id** | Categorical | Feature | Entry point to the hospital | Emergency, Referral, Other |
| **discharge_disposition_id** | Categorical | Feature | Release destination after hospital stay | Discharged to Home, Other |
| **medical_specialty** | Categorical | Feature | Specialty of the admitting physician | Cardiology, Internal Medicine, Emergency, Family/General Practice, Missing, Other |
| **time_in_hospital** | Integer | Feature | Number of days spent in hospital | 1 - 14 |
| **num_lab_procedures** | Integer | Feature | Number of lab tests performed during visit | Numeric Range |
| **num_procedures** | Integer | Feature | Number of non-lab procedures performed during visit | Numeric Range |
| **num_medications** | Integer | Feature | Number of unique generic medications administered | Numeric Range |
| **primary_diagnosis** | Categorical | Feature | Grouped primary diagnosis code | Diabetes, Genitourinary, Musculoskeletal, Respiratory, Other |
| **number_diagnoses** | Integer | Feature | Total number of diagnoses entered into the system | Numeric Range |
| **max_glu_serum** | Categorical | Feature | Glucose serum test results | None, Norm, ">200", ">300" |
| **A1Cresult** | Categorical | Feature | Hemoglobin A1c test results | None, Norm, ">7", ">8" |
| **insulin** | Categorical | Feature | Status of insulin prescription | No, Down, Steady, Up |
| **change** | Binary | Feature | Indicates if there was a change in diabetic medications | No, Ch |
| **diabetesMed** | Binary | Feature | Was any diabetic medication prescribed? | True, False |
| **medicare** | Binary | Feature | Patient has Medicare coverage | True, False |
| **medicaid** | Binary | Feature | Patient has Medicaid coverage | True, False |
| **had_emergency** | Binary | Feature | Patient had emergency room visits in the prior year | True, False |
| **had_inpatient_days** | Binary | Feature | Patient had hospital inpatient days in the prior year | True, False |
| **had_outpatient_days** | Binary | Feature | Patient had outpatient clinic visits in the prior year | True, False |
| **readmit_30_days** | Binary | Target Label | Was the patient re-admitted to the hospital within 30 days? | True, False |

## Fairness Context

In a healthcare setting, hospital readmission prediction models are often used to allocate resources, such as enrolling high-risk patients in post-discharge care management programs. 
If the prediction model suffers from demographic bias—underpredicting readmission risks for specific groups—it leads to **Allocation Harm** where critical post-discharge care is unfairly withheld from historically underserved populations.
