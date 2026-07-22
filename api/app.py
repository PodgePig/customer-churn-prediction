from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

# Load model and feature names at startup
model = joblib.load('model.pkl')
feature_names = joblib.load('feature_names.pkl')  # Save this during preprocessing

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON with customer features.
    Example:
    {
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 846.0,
        "Contract_Month-to-month": 1,
        ...
    }
    """
    try:
        data = request.get_json()
        
        # Convert to DataFrame with proper feature alignment
        input_df = pd.DataFrame([data])
        
        # Ensure all features are present (fill missing with 0)
        for feat in feature_names:
            if feat not in input_df.columns:
                input_df[feat] = 0
        
        # Reorder columns to match training
        input_df = input_df[feature_names]
        
        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0, 1]
        
        return jsonify({
            'churn_prediction': int(prediction),
            'churn_probability': float(probability),
            'message': 'Customer is likely to churn' if prediction == 1 else 'Customer is likely to stay'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Expects JSON with a list of customer records.
    """
    try:
        data = request.get_json()
        if not isinstance(data, list):
            return jsonify({'error': 'Expected a list of records'}), 400
        
        results = []
        for record in data:
            input_df = pd.DataFrame([record])
            for feat in feature_names:
                if feat not in input_df.columns:
                    input_df[feat] = 0
            input_df = input_df[feature_names]
            
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0, 1]
            results.append({
                'churn_prediction': int(pred),
                'churn_probability': float(prob)
            })
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)