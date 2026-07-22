import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import joblib

def run_bayesian_logistic_regression(X_train, y_train, feature_names, n_features_to_use=10):
    """
    Fit a Bayesian Logistic Regression using PyMC with NUTS sampling.
    """
    # Select top N features (use feature importance from XGBoost or just top variance)
    # For simplicity, we'll use a subset of features
    feature_subset = ['tenure', 'MonthlyCharges', 'TotalCharges', 'num_services']
    # Map to actual column names
    available_features = [f for f in feature_subset if f in X_train.columns]
    
    if len(available_features) < 2:
        # Fallback: use first few features
        available_features = X_train.columns[:min(10, len(X_train.columns))].tolist()
    
    print(f"Using features for Bayesian model: {available_features}")
    
    X_bayes = X_train[available_features].values
    y_bayes = y_train.values
    
    n_samples, n_features = X_bayes.shape
    
    with pm.Model() as logistic_model:
        # Priors: weakly informative Normal
        beta = pm.Normal('beta', mu=0, sigma=2, shape=n_features)
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        
        # Linear model
        mu = alpha + pm.math.dot(X_bayes, beta)
        
        # Likelihood: Bernoulli
        theta = pm.Deterministic('theta', pm.math.sigmoid(mu))
        y_obs = pm.Bernoulli('y_obs', p=theta, observed=y_bayes)
        
        # Sample with NUTS
        print("Sampling from posterior with NUTS...")
        trace = pm.sample(
            draws=2000,
            tune=1000,
            chains=4,
            cores=4,
            target_accept=0.9,
            random_seed=42,
            progressbar=True
        )
    
    # Summary
    summary = az.summary(trace, var_names=['alpha', 'beta'])
    print("\nBayesian Model Summary:")
    print(summary)
    
    # Posterior predictive checks
    with logistic_model:
        posterior_predictive = pm.sample_posterior_predictive(
            trace, random_seed=42
        )
    
    # Plot posterior distributions
    az.plot_trace(trace, var_names=['beta'])
    plt.savefig('bayesian_trace.png')
    plt.close()
    
    # Store results
    results = {
        'trace': trace,
        'summary': summary,
        'posterior_predictive': posterior_predictive,
        'features': available_features
    }
    
    joblib.dump(results, "data/bayesian_results.pkl")
    print("Bayesian results saved to data/bayesian_results.pkl")
    
    # Extract posterior means for coefficients
    beta_means = summary.loc[['beta' + str(i) for i in range(n_features)], 'mean'].values
    alpha_mean = summary.loc['alpha', 'mean']
    
    print("\nPosterior mean coefficients:")
    for feat, coef in zip(available_features, beta_means):
        print(f"  {feat}: {coef:.4f}")
    
    return results

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = joblib.load("data/processed_data.pkl")
    run_bayesian_logistic_regression(X_train, y_train, feature_names)