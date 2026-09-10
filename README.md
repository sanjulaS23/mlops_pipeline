# MLOps Pipeline — End to End

A complete MLOps pipeline demonstrating experiment tracking, model training, and data drift monitoring, with automated CI/CD using GitHub Actions.

## What This Project Does

This project trains a machine learning model (Random Forest Classifier), tracks the experiment using MLflow, and monitors incoming data for drift using Evidently AI. Every push to GitHub automatically triggers a pipeline that lints the code, trains the model, and checks for data drift.

## Tech Stack

- **Python** — core programming language
- **MLflow** — experiment tracking and model logging
- **Evidently AI** — data drift detection and reporting
- **scikit-learn** — machine learning model (Random Forest)
- **GitHub Actions** — CI/CD automation
- **flake8** — code linting

## Project Structure
mlops_pipeline/
├── .github/workflows/
│ └── ml_pipeline.yaml # CI/CD pipeline definition
├── src/
│ ├── data/
│ ├── train.py # Trains model, logs to MLflow
│ └── monitor.py # Checks for data drift
├── requirements.txt
└── README.md


## How to Run Locally

1. Clone this repository:
git clone https://github.com/sanjulaS23/mlops_pipeline.git
cd mlops_pipeline


2. Create and activate a virtual environment:
python -m venv venv
.\venv\Scripts\Activate.ps1


3. Install dependencies:

pip install -r requirements.txt


4. Train the model:

python src/train.py


5. View experiment results in MLflow UI:

mlflow ui

   Then open `http://127.0.0.1:5000` in your browser.

6. Run drift monitoring:

python src/monitor.py


## CI/CD Pipeline

Every push to the `main` branch automatically runs:
1. Code linting (flake8)
2. Model training
3. Data drift monitoring
4. Drift report upload as a workflow artifact

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.865 |
| F1 Score | 0.874 |

## Future Improvements

- Data versioning with DVC
- Model registry with automated promotion
- Auto-retraining triggered by drift detection
- Deployment to cloud (AWS/Azure/GCP)