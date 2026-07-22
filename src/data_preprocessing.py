import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath):
    """Load the Telco Churn dataset and perform initial cleaning."""
    df = pd.read_csv(filepath)
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
    df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')
    return df

def engineer_features(df):
    """Feature engineering: tenure bins, interaction terms, etc."""
    df['tenure_bins'] = pd.cut(df['tenure'], 
                               bins=[0, 12, 24, 48, 72, 100], 
                               labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr', '6+yr'])
    df['avg_monthly_charge_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['num_services'] = df[['PhoneService', 'OnlineSecurity', 'OnlineBackup', 
                             'DeviceProtection', 'TechSupport', 'StreamingTV', 
                             'StreamingMovies']].apply(
        lambda x: sum(x == 'Yes'), axis=1
    )
    return df

def preprocess_for_modelling(df):
    """Encode categorical variables and scale numeric features."""
    X = df.drop('Churn', axis=1)
    X = X.drop('tenure_bins', axis=1, errors='ignore')
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    X[categorical_cols] = X[categorical_cols].astype('object')
    
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler, X_encoded.columns

if __name__ == "__main__":
    print("Loading data...")
    df = load_and_clean_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    print("Engineering features...")
    df = engineer_features(df)
    
    print("Preprocessing for modelling...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_for_modelling(df)
    
    print("Saving processed data...")
    joblib.dump((X_train, X_test, y_train, y_test, feature_names), "data/processed_data.pkl")
    joblib.dump(scaler, "data/scaler.pkl")
    
    print(f"Done! Training set: {X_train.shape}, Test set: {X_test.shape}")