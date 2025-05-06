import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from preprocessing import preprocessing_process
from EDA import univariate_analysis, bivariate_analysis, correlation_heatmap
from FeatureEngineering import feature_engineering_pipeline
from imblearn.over_sampling import SMOTE

def run_model_training_pipeline(file_path):
    # Step 1: Preprocessing, EDA, Feature Engineering, SMOTE
    df_cleaned, _, _, _, _ = preprocessing_process(file_path, verbose=False)

    # Optionally call EDA (commented out to suppress plots)
    # univariate_analysis(df_cleaned)
    # bivariate_analysis(df_cleaned)
    # correlation_heatmap(df_cleaned)

    df = feature_engineering_pipeline(file_path)

    drop_cols = ['Exited', 'Surname', 'Geography', 'Gender', 'AgeGroup']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['Exited']

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Step 2: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=0)

    # Step 3: Define Models
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=0),
        'Random Forest': RandomForestClassifier(random_state=0),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    # Step 4: Train and Evaluate
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append([
            name,
            accuracy_score(y_test, y_pred),
            f1_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred)
        ])

    # Step 5: Show Evaluation
    results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])
    print(results_df)

    best_model_name = results_df.sort_values(by='Accuracy', ascending=False).iloc[0]['Model']
    print(f"\nSelected Final Model: {best_model_name}")


# ----------- Run Only If Called Directly -----------
if __name__ == "__main__":
    dataset_path = r"C:\Users\madha\BankChurn\BankChurnProject\Churn_Modelling.csv"
    run_model_training_pipeline(dataset_path)
