import shap
import numpy as np
import pandas as pd
import pickle
import os

from preprocessing import preprocessing_process
from FeatureEngineering import feature_engineering_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from flask import Flask, render_template, request

app = Flask(__name__)  # Correct Flask setup

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

def shap_explanation(file_path, index):
    df_cleaned, _, _, _, _ = preprocessing_process(file_path, verbose=False)
    df = feature_engineering_pipeline(file_path)

    drop_cols = ['Exited', 'Surname', 'Geography', 'Gender', 'AgeGroup']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['Exited']

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=0)

    model = XGBClassifier(random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)

    new_customer = X_test.iloc[index:index + 1]

    explainer = shap.Explainer(model)
    shap_values = explainer(new_customer)

    pred_prob = model.predict_proba(new_customer)[0][1] * 100
    prediction = 'Churn' if pred_prob >= 50 else 'No Churn'

    shap_array = shap_values.values[0].flatten()
    feature_names = new_customer.columns.tolist()

    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_value': shap_array
    })
    shap_df['Impact (%)'] = shap_df['SHAP_value'] / np.sum(np.abs(shap_df['SHAP_value'])) * 100
    shap_df = shap_df.reindex(shap_df['Impact (%)'].abs().sort_values(ascending=False).index)

    shap_list = []
    for i, row in shap_df.head(5).iterrows():
        sign = '+' if row['Impact (%)'] > 0 else '-'
        shap_list.append(f"{row['Feature']}: {sign}{abs(row['Impact (%)']):.1f}%")

    return prediction, pred_prob, shap_list

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "customer_index" in request.form and request.form["customer_index"].strip() != "":
            index = int(request.form["customer_index"])
            file_path = "Churn_Modelling.csv"
            prediction, probability, shap_list = shap_explanation(file_path, index)
        else:
            input_data = {
                'CreditScore': int(request.form["CreditScore"]),
                'Geography': request.form["Geography"],
                'Gender': request.form["Gender"],
                'Age': int(request.form["Age"]),
                'Tenure': int(request.form["Tenure"]),
                'Balance': float(request.form["Balance"]),
                'NumOfProducts': int(request.form["NumOfProducts"]),
                'HasCrCard': int(request.form["HasCrCard"]),
                'EstimatedSalary': float(request.form["EstimatedSalary"])
            }

            prediction, probability, shap_list = predict_churn_for_new_customer(input_data)

        return render_template("result.html", prediction=prediction, probability=f"{probability:.2f}", shap_values=shap_list)

    except Exception as e:
        return f"<h3 style='color:red;'>Error occurred: {str(e)}</h3><a href='/'>Go back</a>"

if __name__ == "__main__":
    app.run(debug=True)
