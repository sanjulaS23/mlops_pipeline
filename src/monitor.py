import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

np.random.seed(42)
reference_data = pd.DataFrame({
    "feature_1": np.random.normal(0, 1, 500),
    "feature_2": np.random.normal(5, 2, 500),
    "feature_3": np.random.uniform(0, 10, 500),
})

current_data = pd.DataFrame({
    "feature_1": np.random.normal(2, 1, 500),
    "feature_2": np.random.normal(9, 2, 500),
    "feature_3": np.random.uniform(0, 10, 500),
})

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)

report.save_html("drift_report.html")

result = report.as_dict()
drift_detected = result["metrics"][0]["result"]["dataset_drift"]

print(f"Drift Detected: {drift_detected}")

if drift_detected:
    print("WARNING: Data drift detected! Model retraining recommended.")
else:
    print("No significant drift detected.")