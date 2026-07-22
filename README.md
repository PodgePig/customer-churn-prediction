# Customer Churn Prediction

An end-to-end machine learning pipeline to predict customer churn for a telecommunications provider. 

This project demonstrates the complete data science lifecycle: exploratory data analysis, statistical hypothesis testing, feature engineering, supervised machine learning with hyperparameter optimisation, Bayesian inference for uncertainty quantification, model explainability, and deployment via a containerised REST API.

## Overview

Customer churn is a critical business metric. This project uses the publicly available IBM Telco Customer Churn dataset to build a robust predictive system that identifies customers at high risk of leaving. The pipeline is structured to mirror a real-world commercial environment, emphasising reproducibility, interpretability, and deployability.

## Key Features

- Statistical Hypothesis Testing: Formal tests (Chi-squared, Mann-Whitney U, Welch's t-test) with effect sizes to validate business intuition.
- Machine Learning: XGBoost, regularised Logistic Regression, and a Stacking Ensemble classifier tuned via Optuna.
- Bayesian Inference: Probabilistic logistic regression using PyMC with NUTS sampling, providing posterior distributions for coefficients.
- Model Explainability: SHAP (SHapley Additive exPlanations) for global feature importance and local, instance-level explanations.
- Deployment: A Flask REST API, containerised with Docker, serving predictions in real-time.
- Reproducibility: Fully version-controlled code with a structured, modular project layout.