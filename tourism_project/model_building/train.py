import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# --- MLOps Configuration ---
# Update this with your actual DagsHub or Remote MLflow URI
# mlflow.set_tracking_uri("https://dagshub.com/Nagaraj4k/VisitWithUs.mlflow") 
mlflow.set_experiment("wellness-tourism-prediction")

api = HfApi()

# --- Data Ingestion ---
# Pointing to the datasets we registered earlier in the pipeline
repo_owner = "Nagaraj4k"
dataset_repo = f"{repo_owner}/VisitWithUs-Dataset"

# Loading datasets directly from Hugging Face Hub
Xtrain = pd.read_csv(f"hf://datasets/{dataset_repo}/Xtrain.csv")
Xtest = pd.read_csv(f"hf://datasets/{dataset_repo}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{dataset_repo}/ytrain.csv")
ytest = pd.read_csv(f"hf://datasets/{dataset_repo}/ytest.csv")

# --- Feature Engineering ---
# Numeric features from your tourism.csv
numeric_features = [
    'Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 
    'PitchSatisfactionScore', 'NumberOfChildrenVisiting', 'MonthlyIncome'
]

# Categorical features requiring encoding
categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 
    'ProductPitched', 'MaritalStatus', 'Designation'
]

# --- Preprocessing & Model Setup ---
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features),
    remainder='passthrough' # Keeps CityTier, Passport, OwnCar
)

# Handle class imbalance for the Wellness Package
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.01, 0.1],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

# --- Training & Experiment Tracking ---
with mlflow.start_run(run_name="XGB_Wellness_Package_Training"):
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring='f1')
    grid_search.fit(Xtrain, ytrain)

    # Log best parameters
    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_
    
    # Custom threshold evaluation
    threshold = 0.45
    y_pred_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    test_report = classification_report(ytest, y_pred, output_dict=True)

    # Log Metrics
    mlflow.log_metrics({
        "test_accuracy": test_report['accuracy'],
        "test_f1-score": test_report['1']['f1-score'],
        "test_recall": test_report['1']['recall']
    })

    # --- Serialization & Artifacts ---
    model_name = "wellness_tourism_model_v1.joblib"
    joblib.dump(best_model, model_name)
    mlflow.log_artifact(model_name, artifact_path="model")

    # --- Hugging Face Model Registry ---
    model_repo_id = f"{repo_owner}/Wellness-Tourism-Model"
    
    try:
        api.repo_info(repo_id=model_repo_id, repo_type="model")
    except RepositoryNotFoundError:
        create_repo(repo_id=model_repo_id, repo_type="model", private=False)

    api.upload_file(
        path_or_fileobj=model_name,
        path_in_repo=model_name,
        repo_id=model_repo_id,
        repo_type="model"
    )
    print(f"Model successfully deployed to HF: {model_repo_id}")
