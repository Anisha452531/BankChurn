import os
from preprocessing import preprocessing_process
from FeatureEngineering import feature_engineering_pipeline
from imblearn.over_sampling import SMOTE

def run_smote_pipeline(file_path):
    # Step 1: Preprocess and Feature Engineer the Data
    df = feature_engineering_pipeline(file_path)

    # Step 2: Prepare X and y for SMOTE
    drop_cols = ['Exited', 'Surname', 'Geography', 'Gender', 'AgeGroup']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['Exited']

    # Step 3: Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Step 4: Output Result
    print("✅ SMOTE applied successfully.")
    print("Class distribution after SMOTE:")
    print(y_resampled.value_counts())

if __name__ == "__main__":
    dataset_path = r"BankChurnProject\Churn_Modelling.csv"
    run_smote_pipeline(dataset_path)
