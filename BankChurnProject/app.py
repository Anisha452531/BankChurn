import shap
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from xgboost import XGBClassifier
from flask import Flask, render_template, request
from NewChurnPrediction import predict_churn_for_new_customer  # Optional: if needed for new customer prediction

app = Flask(__name__)

# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocessing_process(file_path='Churn_Modelling.csv', verbose=False):
    df = pd.read_csv(file_path)

    if verbose:
        print("Initial Dataset Preview:\n", df.head())
        print("Shape of the dataset:", df.shape)
        print("\nDataset Info:")
        print(df.info())

    df = df.drop_duplicates()

    for col in ['CustomerId', 'RowNumber']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    le_gender = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])

    le_geo = LabelEncoder()
    df['Geography'] = le_geo.fit_transform(df['Geography'])

    minmax_scaler = MinMaxScaler()
    df[['EstimatedSalary', 'Balance']] = minmax_scaler.fit_transform(df[['EstimatedSalary', 'Balance']])

    std_scaler = StandardScaler()
    df[['NumOfProducts']] = std_scaler.fit_transform(df[['NumOfProducts']])

    if verbose:
        print("\nPreprocessed Dataset Preview:\n", df.head())

    return df

# -------------------------------
# Feature Engineering Function
# -------------------------------
def feature_engineering_pipeline(file_path='Churn_Modelling.csv', verbose=False):
    df_cleaned = preprocessing_process(file_path, verbose=False)
    df = df_cleaned.copy()

    df['Income_Balance_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)

    def age_group(age):
        if age < 30:
            return 'Young'
        elif age < 50:
            return 'Middle'
        else:
            return 'Old'

    df['AgeGroup'] = df['Age'].apply(age_group)
    age_mapping = {'Young': 0, 'Middle': 1, 'Old': 2}
    df['AgeGroup_Encoded'] = df['AgeGroup'].map(age_mapping)

    df['Income_Age_Interaction'] = df['EstimatedSalary'] * df['Age']
    df['Spending_vs_Income'] = df['Balance'] / (df['EstimatedSalary'] + 1)

    return df

# -------------------------------
# SHAP Explanation Function
# -------------------------------
def shap_explanation(file_path, index):
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

    shap_array = shap_values.values[0]
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

# -------------------------------
# Flask Routes
# -------------------------------
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
