import shap
import numpy as np
import pandas as pd
from preprocessing import preprocessing_process
from FeatureEngineering import feature_engineering_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from flask import Flask, render_template, request
from NewChurnPrediction import predict_churn_for_new_customer  # Keep if you want to keep new customer prediction separate

app = Flask(__name__)

def shap_explanation(file_path, index):
    # Preprocess and feature engineer
    df_cleaned, _, _, _, _ = preprocessing_process(file_path, verbose=False)
    df = feature_engineering_pipeline(file_path)

    # Define features and target
    drop_cols = ['Exited', 'Surname', 'Geography', 'Gender', 'AgeGroup']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['Exited']

    # Handle imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=0)

    # Train model
    model = XGBClassifier(random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Select customer instance
    new_customer = X_test.iloc[index:index + 1]

    # SHAP explanation
    explainer = shap.Explainer(model)
    shap_values = explainer(new_customer)

    # Predict probability
    pred_prob = model.predict_proba(new_customer)[0][1] * 100
    prediction = 'Churn' if pred_prob >= 50 else 'No Churn'

    # Prepare SHAP impact DataFrame
    shap_array = shap_values.values[0]
    feature_names = new_customer.columns.tolist()
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_value': shap_array
    })
    shap_df['Impact (%)'] = shap_df['SHAP_value'] / np.sum(np.abs(shap_df['SHAP_value'])) * 100
    shap_df = shap_df.reindex(shap_df['Impact (%)'].abs().sort_values(ascending=False).index)

    # Create output summary
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
        # Case 1: Predict by existing customer index
        if "customer_index" in request.form and request.form["customer_index"].strip() != "":
            index = int(request.form["customer_index"])
            file_path = "Churn_Modelling.csv"
            prediction, probability, shap_list = shap_explanation(file_path, index)

        # Case 2: Predict for a new customer
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

        # Render result page
        return render_template("result.html", prediction=prediction, probability=f"{probability:.2f}", shap_values=shap_list)

    except Exception as e:
        # Error handling: Show error on the same page
        return f"<h3 style='color:red;'>Error occurred: {str(e)}</h3><a href='/'>Go back</a>"

if __name__ == "__main__":
    app.run(debug=True)
