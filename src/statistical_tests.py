import pandas as pd
import numpy as np
from scipy import stats
from src.data_preprocessing import load_and_clean_data

def run_statistical_tests(df):
    """
    Perform statistical tests to understand churn drivers.
    """
    results = {}
    
    # Convert Churn to binary for tests
    df['churn_binary'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # ---------- Chi-squared test: Contract type vs Churn ----------
    contingency_table = pd.crosstab(df['Contract'], df['Churn'])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    results['chi2_contract'] = {
        'chi2': chi2,
        'p_value': p_val,
        'dof': dof,
        'significant': p_val < 0.01
    }
    print(f"Chi-squared test (Contract vs Churn): χ²={chi2:.4f}, p={p_val:.4e}")
    print(f"  → {'Significant' if p_val < 0.01 else 'Not significant'} at α=0.01")
    
    # ---------- Mann-Whitney U test: MonthlyCharges vs Churn ----------
    churn_charges = df[df['churn_binary'] == 1]['MonthlyCharges']
    non_churn_charges = df[df['churn_binary'] == 0]['MonthlyCharges']
    u_stat, p_val_mw = stats.mannwhitneyu(churn_charges, non_churn_charges, alternative='two-sided')
    results['mannwhitney_monthly'] = {
        'u_stat': u_stat,
        'p_value': p_val_mw,
        'significant': p_val_mw < 0.01
    }
    print(f"Mann-Whitney U (MonthlyCharges vs Churn): U={u_stat:.4f}, p={p_val_mw:.4e}")
    print(f"  → {'Significant' if p_val_mw < 0.01 else 'Not significant'} at α=0.01")
    
    # ---------- Additional: t-test for tenure (checking normality assumption) ----------
    # Tenure is roughly normal, but we'll use Welch's t-test
    churn_tenure = df[df['churn_binary'] == 1]['tenure']
    non_churn_tenure = df[df['churn_binary'] == 0]['tenure']
    t_stat, p_val_t = stats.ttest_ind(churn_tenure, non_churn_tenure, equal_var=False)
    results['ttest_tenure'] = {
        't_stat': t_stat,
        'p_value': p_val_t,
        'significant': p_val_t < 0.01
    }
    print(f"Welch's t-test (tenure vs Churn): t={t_stat:.4f}, p={p_val_t:.4e}")
    
    # ---------- Effect sizes ----------
    # Cramer's V for chi-squared
    n = contingency_table.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
    results['cramers_v_contract'] = cramers_v
    print(f"Cramer's V (Contract vs Churn): {cramers_v:.4f}")
    
    # Cohen's d for MonthlyCharges
    pooled_std = np.sqrt(((len(churn_charges) - 1) * churn_charges.var() + 
                          (len(non_churn_charges) - 1) * non_churn_charges.var()) / 
                         (len(churn_charges) + len(non_churn_charges) - 2))
    cohens_d = (churn_charges.mean() - non_churn_charges.mean()) / pooled_std
    results['cohens_d_monthly'] = cohens_d
    print(f"Cohen's d (MonthlyCharges): {cohens_d:.4f}")
    
    return results

if __name__ == "__main__":
    df = load_and_clean_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    results = run_statistical_tests(df)