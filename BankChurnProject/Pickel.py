import pickle
import os

def save_model_artifacts(
    model, minmax_scaler, std_scaler, le_gender, le_geo, X_columns,
    output_dir="artifacts"
):
    """
    Saves model and preprocessing artifacts to disk using pickle.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'final_model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    with open(os.path.join(output_dir, 'minmax_scaler.pkl'), 'wb') as f:
        pickle.dump(minmax_scaler, f)

    with open(os.path.join(output_dir, 'std_scaler.pkl'), 'wb') as f:
        pickle.dump(std_scaler, f)

    with open(os.path.join(output_dir, 'le_gender.pkl'), 'wb') as f:
        pickle.dump(le_gender, f)

    with open(os.path.join(output_dir, 'le_geo.pkl'), 'wb') as f:
        pickle.dump(le_geo, f)

    with open(os.path.join(output_dir, 'X_columns.pkl'), 'wb') as f:
        pickle.dump(X_columns, f)

    print(f"All model artifacts saved to '{output_dir}' successfully.")

# ✅ Function call added below (for testing purpose)
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
    import pandas as pd

    # Dummy DataFrame
    df = pd.DataFrame({
        'CreditScore': [600, 700],
        'Geography': ['France', 'Germany'],
        'Gender': ['Male', 'Female'],
        'Age': [30, 40],
        'Tenure': [3, 7],
        'Balance': [50000, 70000],
        'NumOfProducts': [1, 2],
        'HasCrCard': [1, 0],
        'EstimatedSalary': [40000, 60000],
        'Exited': [0, 1]
    })

    X = df.drop('Exited', axis=1)
    y = df['Exited']

    le_gender = LabelEncoder()
    le_geo = LabelEncoder()

    X['Gender'] = le_gender.fit_transform(X['Gender'])
    X['Geography'] = le_geo.fit_transform(X['Geography'])

    minmax_scaler = MinMaxScaler()
    std_scaler = StandardScaler()

    X[['EstimatedSalary', 'Balance']] = minmax_scaler.fit_transform(X[['EstimatedSalary', 'Balance']])
    X[['NumOfProducts']] = std_scaler.fit_transform(X[['NumOfProducts']])

    model = RandomForestClassifier()
    model.fit(X, y)

    # 🎯 Call the function with actual objects
    save_model_artifacts(
        model=model,
        minmax_scaler=minmax_scaler,
        std_scaler=std_scaler,
        le_gender=le_gender,
        le_geo=le_geo,
        X_columns=X.columns.tolist()
    )
