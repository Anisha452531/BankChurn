import pickle
import pandas as pd
import shap
import os

def predict_churn_for_new_customer():
    # Load saved assets from 'artifacts' folder
    base_path = "artifacts"
    final_model = pickle.load(open(os.path.join(base_path, 'final_model.pkl'), 'rb'))
    minmax_scaler = pickle.load(open(os.path.join(base_path, 'minmax_scaler.pkl'), 'rb'))
    std_scaler = pickle.load(open(os.path.join(base_path, 'std_scaler.pkl'), 'rb'))
    le_gender = pickle.load(open(os.path.join(base_path, 'le_gender.pkl'), 'rb'))
    le_geo = pickle.load(open(os.path.join(base_path, 'le_geo.pkl'), 'rb'))
    X_columns = pickle.load(open(os.path.join(base_path, 'X_columns.pkl'), 'rb'))

    # Get input from user
    print("Enter New Customer Details:")
    credit_score = int(input("Credit Score: "))
    geography = input("Geography (France/Germany/Spain): ")
    gender = input("Gender (Male/Female): ")
    age = int(input("Age: "))
    tenure = int(input("Tenure: "))
    balance = float(input("Balance: "))
    num_of_products = int(input("Num of Products: "))
    has_cr_card = int(input("Has Credit Card (1 for Yes, 0 for No): "))
    estimated_salary = float(input("Estimated Salary: "))

    # Create DataFrame
    new_customer = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'EstimatedSalary': [estimated_salary]
    })

    # Label Encoding
    new_customer['Gender'] = le_gender.transform(new_customer['Gender'])
    new_customer['Geography'] = le_geo.transform(new_customer['Geography'])

    # Apply scalers
    new_customer[['EstimatedSalary', 'Balance']] = minmax_scaler.transform(new_customer[['EstimatedSalary', 'Balance']])
    new_customer[['NumOfProducts']] = std_scaler.transform(new_customer[['NumOfProducts']])

    # Align with training columns
    new_customer = new_customer.reindex(columns=X_columns, fill_value=0)

    # Prediction
    pred_prob = final_model.predict_proba(new_customer)[0][1] * 100
    prediction = 'Churn' if pred_prob >= 50 else 'No Churn'
    print(f"\nPrediction: {prediction}")
    print(f"Prediction Probability: {pred_prob:.2f}%")

    # SHAP Explanation
    explainer = shap.Explainer(final_model, feature_names=X_columns)
    shap_values = explainer(new_customer)

    shap_df = pd.DataFrame({
        'Feature': X_columns,
        'SHAP Value': shap_values.values[0, :, 1]
    }).sort_values(by='SHAP Value', key=abs, ascending=False)

    print("\nTop Features Contributing to Churn:")
    for i in range(min(7, len(shap_df))):
        feature = shap_df.iloc[i]
        impact = '+' if feature['SHAP Value'] > 0 else '-'
        print(f"- {feature['Feature']}: {impact}{abs(feature['SHAP Value']) * 100:.1f}%")

# Call the function
if __name__ == "__main__":
    predict_churn_for_new_customer()
