import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import preprocessing_process  # Import from preprocessing.py

# ----------------------------
# Univariate Analysis
# ----------------------------
def univariate_analysis(df):
    # Age Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Age'], kde=True).set(title='Age Distribution')
    plt.show()

    # Estimated Salary Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['EstimatedSalary'], kde=True).set(title='Estimated Salary')
    plt.show()

    # Balance Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Balance'], kde=True).set(title='Balance')
    plt.show()

    # Num of Products Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['NumOfProducts'], kde=True).set(title='Num of Products')
    plt.show()

    # Geography Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Geography', data=df).set(title='Geography Distribution')
    plt.show()

    # Gender Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Gender', data=df).set(title='Gender Distribution')
    plt.show()

# ----------------------------
# Bivariate Analysis
# ----------------------------
def bivariate_analysis(df):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='EstimatedSalary', y='Balance', hue='Exited', data=df)
    plt.title('Salary vs Balance vs Churn')
    plt.show()

# ----------------------------
# Correlation Heatmap
# ----------------------------
def correlation_heatmap(df):
    plt.figure(figsize=(12, 6))
    numerical_df = df.select_dtypes(include=['number'])
    sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    # Get absolute path to the dataset
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, 'Churn_Modelling.csv')  # Ensure this file exists

    # Preprocess dataset without showing internal output
    df_cleaned, _, _, _, _ = preprocessing_process(file_path, verbose=False)

    # Run EDA
    univariate_analysis(df_cleaned)
    bivariate_analysis(df_cleaned)
    correlation_heatmap(df_cleaned)
