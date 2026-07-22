import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

def explain_with_shap(model, X_train, X_test, feature_names):
    """
    Generate SHAP explanations for the trained model.
    """
    print("Generating SHAP explanations...")
    
    # For tree-based models (XGBoost)
    if isinstance(model, xgb.XGBClassifier):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Summary plot - global feature importance
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Bar plot - mean absolute SHAP values
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                          plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig('shap_bar_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Waterfall plot for a single prediction (first test sample)
        plt.figure(figsize=(12, 6))
        shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                             base_values=explainer.expected_value,
                                             data=X_test.iloc[0].values,
                                             feature_names=feature_names), show=False)
        plt.tight_layout()
        plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save SHAP values for later
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_df.to_csv('shap_values.csv', index=False)
        
        print("SHAP plots saved: shap_summary_plot.png, shap_bar_plot.png, shap_waterfall.png")
        print("SHAP values saved to shap_values.csv")
        
        return shap_values, explainer
    
    else:
        print("SHAP only supported for tree-based models. Use KernelExplainer for other models.")
        return None, None

if __name__ == "__main__":
    # Load data and model
    X_train, X_test, y_train, y_test, feature_names = joblib.load("data/processed_data.pkl")
    model = joblib.load("api/model.pkl")
    
    # If model is a stacking ensemble, extract the XGBoost component
    if hasattr(model, 'estimators_'):
        # Extract XGBoost from the stack
        xgb_model = None
        for name, est in model.named_estimators_.items():
            if 'xgb' in name.lower() or isinstance(est, xgb.XGBClassifier):
                xgb_model = est
                break
        if xgb_model is None:
            print("Warning: Could not find XGBoost in ensemble. Using ensemble directly (may be slow).")
            xgb_model = model
    else:
        xgb_model = model
    
    shap_values, explainer = explain_with_shap(xgb_model, X_train, X_test, feature_names)