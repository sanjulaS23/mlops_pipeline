import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Step 1: Create "reference" data (original training data)
np.random.seed(42)
reference_data = pd.DataFrame({
    "feature_1": np.random.normal(0, 1, 500),
    "feature_2": np.random.normal(5, 2, 500),
    "feature_3": np.random.uniform(0, 10, 500),
})

# Step 2: Create "current" data (simulating drift by shifting the distribution)
current_data = pd.DataFrame({
    "feature_1": np.random.normal(2, 1, 500),    # mean shifted from 0 to 2 (drift!)
    "feature_2": np.random.normal(9, 2, 500),    # mean shifted from 5 to 9 (drift!)
    "feature_3": np.random.uniform(0, 10, 500),  # no drift
})

# Step 3: Generate a drift report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)

# Step 4: Save the report as an HTML file
report.save_html("drift_report.html")

# Step 5: Extract drift result as a dictionary (so code can react to it)
result = report.as_dict()
drift_detected = result["metrics"][0]["result"]["dataset_drift"]

print(f"Drift Detected: {drift_detected}")

if drift_detected:
    print("⚠️  WARNING: Data drift detected! Model retraining recommended.")
else:
    print("✅ No significant drift detected.")
