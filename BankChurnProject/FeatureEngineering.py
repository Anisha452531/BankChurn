import os
import pandas as pd
from preprocessing import preprocessing_process
from EDA import univariate_analysis, bivariate_analysis, correlation_heatmap

# -----------------------
# Feature Engineering Function
# -----------------------
def feature_engineering_pipeline(file_path='C:/Users/anish/BankChurn/BankChurnProject/Churn_Modelling.csv'):
    
    df_cleaned, _, _, _, _ = preprocessing_process(file_path, verbose=False)
    
    df = df_cleaned.copy()  

    # ------------------
    # 1. Income to Balance Ratio
    # -----------------------
    df['Income_Balance_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)

    # -----------------------
    # 2. Age Group
    # ------------------
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

    # -----------------------
    # 3. Income-Age Interaction
    # -----------------------
    df['Income_Age_Interaction'] = df['EstimatedSalary'] * df['Age']

    # -----------------------
    # 4. Spending vs Income
    # -----------------------
    df['Spending_vs_Income'] = df['Balance'] / (df['EstimatedSalary'] + 1)

    
    print("Feature Engineered Data Preview:\n", df.head())

    return df  



if __name__ == "__main__":
    final_df = feature_engineering_pipeline('C:/Users/anish/BankChurn/BankChurnProject/Churn_Modelling.csv')
