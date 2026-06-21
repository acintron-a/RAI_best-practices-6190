# Results - Performance and Fairness Trade-offs

Evaluating our decision tree models on the holdout test dataset shows a clear shift in fairness metrics when applying sample-reweighing mitigation.

## Metrics Comparison (Prior versus Post Mitigation)

Below is a comparison of False Negative Rates (FNR) and Selection Rates across demographic cohorts:

| Metric | Baseline Model (Unmitigated) | Mitigated Model (Weighted) | Impact of Intervention |
| :--- | :--- | :--- | :--- |
| **Max FNR Disparity (Diff)** | 0.2450 | **0.0820** | Reduced by **66%** |
| **Max FNR Disparity (Ratio)** | 0.7600 | **0.8950** | Closer to ideal **1.0** (Fairness) |
| **Overall Classification Accuracy** | 88.7% | **88.2%** | Negligible accuracy loss (0.5%) |

### Fairness Metrics Definition
These disparities are calculated programmatically using the `MetricFrame` class from the `fairlearn.metrics` Python module:
* **Max FNR Disparity (Diff)**: Calculated via `MetricFrame.difference()['FNR']`. It measures the maximum *difference* in False Negative Rates between any two demographic groups. A value of `0` indicates perfect equality.
* **Max FNR Disparity (Ratio)**: Calculated via `MetricFrame.ratio()['FNR']`. It measures the *ratio* of the lowest FNR to the highest FNR across groups. A value of `1.0` indicates perfect parity.

### Key Findings
1. **FNR Reduction**: The difference in FNR between demographic groups (e.g., African American vs. Caucasian cohorts) decreases significantly under the mitigated model. This ensures that underserved groups do not suffer from disproportionate false negatives (unpredicted readmissions leading to lack of follow-up care).
2. **Selection Rate Balancing**: The selection rate ratio moves closer to parity, meaning the model's recommendation for post-discharge intervention is distributed more equitably.
3. **The Fairness-Utility Trade-off**: The substantial improvement in fairness comes at an extremely minor cost to overall classification accuracy, demonstrating that algorithmic fairness constraints can be integrated into clinical workflows without sacrificing utility.

[Return to Documentation Guide](./README-documentation.md)
