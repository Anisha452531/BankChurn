import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

def preprocessing_process(file_path='BankChurnProject/Churn_Modelling.csv', verbose=True):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    if verbose:
        print("Initial Dataset Preview:\n", df.head())
        print("Shape of the dataset:", df.shape)
        print("\nDataset Info:")
        print(df.info())
    
    # Check for missing values
    if verbose:
        print("\nMissing Values:\n", df.isnull().sum())
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Drop irrelevant columns
    for col in ['CustomerId', 'RowNumber']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    # Encode categorical features
    le_gender = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])

    le_geo = LabelEncoder()
    df['Geography'] = le_geo.fit_transform(df['Geography'])

    # Scale features
    minmax_scaler = MinMaxScaler()
    df[['EstimatedSalary', 'Balance']] = minmax_scaler.fit_transform(df[['EstimatedSalary', 'Balance']])
    
    std_scaler = StandardScaler()
    df[['NumOfProducts']] = std_scaler.fit_transform(df[['NumOfProducts']])
    
    if verbose:
        print("\nPreprocessed Dataset Preview:\n", df.head())
    
    return df, le_gender, le_geo, minmax_scaler, std_scaler

# Calling the function inside the main block
if __name__ == "__main__":
    df_cleaned, le_gender, le_geo, minmax_scaler, std_scaler = preprocessing_process(verbose=True)
    print("Final cleaned data shape:", df_cleaned.shape)
