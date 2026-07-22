import pandas as pd
import numpy as np
import joblib
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def objective_xgb(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for XGBoost hyperparameter tuning."""
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 3.0),
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict(X_val)
    return 1 - accuracy_score(y_val, y_pred)  # Minimize error

def train_models(X_train, X_test, y_train, y_test):
    """
    Train XGBoost, Logistic Regression, and a stacked ensemble.
    """
    # Split training into train/val for Optuna
    from sklearn.model_selection import train_test_split
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # ---------- 1. XGBoost with Optuna ----------
    print("Optimizing XGBoost with Optuna...")
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective_xgb(trial, X_train_sub, y_train_sub, X_val, y_val), 
                   n_trials=50, show_progress_bar=True)
    
    best_params = study.best_params
    print(f"Best XGBoost params: {best_params}")
    
    xgb_model = xgb.XGBClassifier(**best_params, eval_metric='logloss', 
                                   use_label_encoder=False, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    
    # ---------- 2. Logistic Regression ----------
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_model.fit(X_train, y_train)
    
    # ---------- 3. Stacked Ensemble ----------
    print("\nTraining Stacked Ensemble...")
    estimators = [
        ('xgb', xgb_model),
        ('lr', lr_model)
    ]
    stack_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5
    )
    stack_model.fit(X_train, y_train)
    
    # ---------- Evaluate ----------
    models = {
        'XGBoost': xgb_model,
        'LogisticRegression': lr_model,
        'StackingEnsemble': stack_model
    }
    
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        results[name] = {
            'accuracy': acc,
            'roc_auc': auc,
            'model': model
        }
        
        print(f"\n{name}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  ROC-AUC: {auc:.4f}")
        print(f"  Classification Report:\n{classification_report(y_test, y_pred)}")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    # Save the best model (stacking ensemble usually wins)
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    print(f"\nBest model: {best_model_name} with ROC-AUC={results[best_model_name]['roc_auc']:.4f}")
    
    joblib.dump(results[best_model_name]['model'], "api/model.pkl")
    print("Model saved to api/model.pkl")
    
    return results

if __name__ == "__main__":
    # Load processed data
    X_train, X_test, y_train, y_test, feature_names = joblib.load("data/processed_data.pkl")
    results = train_models(X_train, X_test, y_train, y_test)