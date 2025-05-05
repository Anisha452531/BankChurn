#Loading of DataSet
import pandas as pd
df = pd.read_csv("Churn_Modelling.csv")
print(df.head())
print("Shape of the dataset:",df.shape)
print(df.info())
#Data Preprocessing
#1. No Missing Values
print("Missing Values:\n", df.isnull().sum())
#2. Remove Duplicates
df = df.drop_duplicates()
#3. Drop Irrelevant Columns
if 'CustomerId' in df.columns:
    df.drop('CustomerId', axis=1, inplace=True)
if 'RowNumber' in df.columns:
    df.drop('RowNumber', axis=1, inplace=True)
print(df.head())

#4. Encoding Categorical Features
from sklearn.preprocessing import LabelEncoder
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])
le_geo = LabelEncoder()
df['Geography'] = le_geo.fit_transform(df['Geography'])
print (df.head())
#5. Scaling and Standardization
from sklearn.preprocessing import MinMaxScaler, StandardScaler

minmax_scaler = MinMaxScaler()
df[['EstimatedSalary', 'Balance']] = minmax_scaler.fit_transform(df[['EstimatedSalary', 'Balance']]) # Changed 'dataset' to 'df'

std_scaler = StandardScaler()
df[['NumOfProducts']] = std_scaler.fit_transform(df[['NumOfProducts']])
print(df.head())
