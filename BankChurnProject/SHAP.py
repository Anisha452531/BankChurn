import shap
import numpy as np
import pandas as pd
from preprocessing import preprocessing_process
from FeatureEngineering import feature_engineering_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Step 1: Define a function for SHAP individual customer explanation
def shap_explanation(file_path, index):
    # Step 2: Preprocess, feature engineer, and split data
    df_cleaned, _, _, _, _ = preprocessing_process(file_path, verbose=False)
    df = feature_engineering_pipeline(file_path)

    drop_cols = ['Exited', 'Surname', 'Geography', 'Gender', 'AgeGroup']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['Exited']

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=0)

    # Step 3: Train the model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Step 4: Select the customer instance based on the user input index
    new_customer = X_test.iloc[index:index + 1]

    # Step 5: Initialize SHAP TreeExplainer
    explainer = shap.Explainer(model)

    # Step 6: Compute SHAP values
    shap_values = explainer(new_customer)
    
    # Debugging: Check the shape of SHAP values
    print(f"SHAP values shape: {shap_values.values.shape}")

    # Step 7: Get predicted probability for churn
    pred_prob = model.predict_proba(new_customer)[0][1] * 100

    # Step 8: Check if SHAP values are 1D or 2D and access accordingly
    if shap_values.values.ndim == 2:
        shap_array = shap_values.values[0, :]  # Correct index for 2D array
    else:
        shap_array = shap_values.values  # For 1D array, directly use the values

    # Step 9: Feature names
    feature_names = new_customer.columns.tolist()

    # Step 10: Create DataFrame for formatted explanation
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_value': shap_array
    })

    # Step 11: Compute percentage impact of each feature
    shap_df['Impact (%)'] = shap_df['SHAP_value'] / np.sum(np.abs(shap_df['SHAP_value'])) * 100

    # Step 12: Sort by absolute impact
    shap_df = shap_df.reindex(shap_df['Impact (%)'].abs().sort_values(ascending=False).index)

    # Step 13: Display the result (without any previous print statements)
    print(f"Prediction: {'Churn' if pred_prob >= 50 else 'No Churn'}")
    print(f"Prediction Probability: {pred_prob:.2f}%\n")

    print("Top Features Contributing to Churn:")
    for i, row in shap_df.head(5).iterrows():
        sign = '+' if row['Impact (%)'] > 0 else ''
        print(f"- {row['Feature']}: {sign}{row['Impact (%)']:.1f}%")

# ----------- Run Only If Called Directly -----------    
if __name__ == "__main__":
    # Get the input index from the user
    user_index = int(input("Enter the index of the customer you want to analyze (e.g., 0): "))
    
    # Define dataset path
    dataset_path = r"C:\Users\anish\BankChurn\BankChurnProject\Churn_Modelling.csv"
    
    # Call the function
    shap_explanation(dataset_path, user_index)
