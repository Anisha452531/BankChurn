import pickle
import os
import pandas as pd
import shap
import numpy as np

def predict_churn_for_new_customer(input_data):
    base_path = "artifacts"
    
    # Load models and encoders
    final_model = pickle.load(open(os.path.join(base_path, 'final_model.pkl'), 'rb'))
    minmax_scaler = pickle.load(open(os.path.join(base_path, 'minmax_scaler.pkl'), 'rb'))
    std_scaler = pickle.load(open(os.path.join(base_path, 'std_scaler.pkl'), 'rb'))
    le_gender = pickle.load(open(os.path.join(base_path, 'le_gender.pkl'), 'rb'))
    le_geo = pickle.load(open(os.path.join(base_path, 'le_geo.pkl'), 'rb'))
    X_columns = pickle.load(open(os.path.join(base_path, 'X_columns.pkl'), 'rb'))

    # Create DataFrame from input
    new_customer = pd.DataFrame([input_data])

    # Encode categorical variables
    new_customer['Gender'] = le_gender.transform(new_customer['Gender'].astype(str))
    new_customer['Geography'] = le_geo.transform(new_customer['Geography'].astype(str))

    # Dummy row to scale with original scaler
    dummy = pd.DataFrame([[0] * minmax_scaler.n_features_in_], columns=minmax_scaler.feature_names_in_)
    dummy.loc[0, 'EstimatedSalary'] = new_customer['EstimatedSalary'].values[0]
    dummy.loc[0, 'Balance'] = new_customer['Balance'].values[0]

    scaled = minmax_scaler.transform(dummy)
    scaled_df = pd.DataFrame(scaled, columns=minmax_scaler.feature_names_in_)

    new_customer['EstimatedSalary'] = scaled_df['EstimatedSalary'].values
    new_customer['Balance'] = scaled_df['Balance'].values

    # Standard scale NumOfProducts
    new_customer['NumOfProducts'] = std_scaler.transform(
        new_customer[['NumOfProducts']]
    )[:, 0]

    # Align with model input features
    new_customer = new_customer.reindex(columns=X_columns, fill_value=0)

    # Predict churn probability
    pred_prob = final_model.predict_proba(new_customer)[0][1] * 100
    prediction = 'Churn' if pred_prob >= 50 else 'No Churn'

    # SHAP Explanation
    explainer = shap.Explainer(final_model, feature_names=X_columns)
    shap_values = explainer(new_customer)

    shap_array = shap_values.values[0].flatten()

    # Fix: ensure both arrays are same length
    min_len = min(len(shap_array), len(X_columns))
    shap_df = pd.DataFrame({
        'Feature': X_columns[:min_len],
        'SHAP Value': shap_array[:min_len]
    }).sort_values(by='SHAP Value', key=abs, ascending=False)

    # Prepare top features
    shap_list = []
    for i in range(min(5, len(shap_df))):
        feature = shap_df.iloc[i]
        sign = '+' if feature['SHAP Value'] > 0 else '-'
        shap_list.append(f"{feature['Feature']}: {sign}{abs(feature['SHAP Value']) * 100:.1f}%")

    return prediction, pred_prob, shap_list
