# 🏨 Visit with Us: Wellness Tourism Prediction Pipeline

![MLOps Pipeline](https://img.shields.io/badge/MLOps-CI%2FCD-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Streamlit-orange)

## 📋 Project Objective
"Visit with Us" is a travel company aiming to optimize the sales of its newly introduced **Wellness Tourism Package**. This repository contains an end-to-end MLOps pipeline designed to predict customer purchase behavior. By identifying high-potential customers before contact, we empower the marketing team to drive business growth and operational efficiency.

## 🏗️ System Architecture
The pipeline automates the entire machine learning lifecycle:
1. **Data Ingestion:** Automated retrieval of customer interaction data.
2. **Preprocessing:** Handling missing values, scaling numeric data, and encoding categorical variables (Age, Designation, Passport status, etc.).
3. **Model Building:** Training an optimized **XGBoost/Random Forest** classifier with hyperparameter tuning.
4. **Experiment Tracking:** Logging metrics (Accuracy, F1-Score, Recall) using **MLflow**.
5. **Deployment:** Continuous Deployment (CD) to **Hugging Face Spaces** via Docker.

## 📁 Project Structure
```text
VisitWithUs/
├── .github/workflows/       # GitHub Actions (CI/CD Pipeline)
│   └── main.yml             # Automation script
├── tourism_project/
│   ├── data/                # Cleaned datasets
│   ├── model_building/      # Training & Registration scripts
│   │   ├── train.py
│   │   └── data_register.py
│   └── deployment/          # Deployment & Hosting files
│       ├── app.py           # Streamlit UI
│       ├── Dockerfile       # Container configuration
│       ├── hosting.py       # HF Spaces push script
│       └── requirements.txt # Project dependencies
├── tourism.csv              # Source data
└── README.md
