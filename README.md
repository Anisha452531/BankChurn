# 🔍 Customer Churn Prediction Using Machine Learning

This project is a machine learning-based web application that predicts customer churn. By analyzing a dataset of 10,000 customers, the system uncovers hidden behavioral patterns and helps businesses make informed retention decisions. The tool not only analyzes existing customer records but also allows for real-time prediction by entering new customer data.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation Guide](#installation-guide)
- [Usage Guide](#usage-guide)
- [Project Architecture](#project-architecture)
- [Technologies Used](#technologies-used)
- [Dependencies](#dependencies)
- [License](#license)
- [Authors](#authors)

---

## Project Overview

Goal: Predict if a customer will churn (leave the service) based on demographic, financial, and behavioral data using machine learning.

This project uses:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Imbalanced learning (SMOTE)
- Classification models like Random Forest, Decision Tree, and XGBoost
- SHAP for model explainability
- Flask for web interface
- Git & GitHub for version control and collaboration

---

## Dataset

- Filename: churn_modeling.csv  
- Records: 10,000 customers  
- Columns: Customer ID, Geography, Gender, Age, Balance, Tenure, IsActiveMember, EstimatedSalary, and more  
- Target: Exited column (1 = churned, 0 = retained)

---

## Features

- Predict churn for an existing customer using ID  
- Predict churn status for new customer data  
- SMOTE for handling class imbalance  
- SHAP values for visual interpretation of model decisions  
- Dynamic EDA visualizations (univariate, bivariate, correlation)  
- Trained models with evaluation metrics (F1, Accuracy, Precision, Recall)  
- Clean and interactive frontend built with HTML/CSS/JS

---

## Installation Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/Anisha452531/BankChurn.git
cd BankChurn
```
 ---
## Usage Guide

To predict churn for an existing customer, enter the Customer ID in the search box.

To predict churn for a new customer, fill in the customer details form.

The system will display whether the customer is likely to churn along with detailed prediction insights.

Visualizations of data patterns and explanations are also available for better understanding.

---

## Project Architecture

- Data Layer: Contains the dataset churn_modeling.csv
- Preprocessing: Python scripts for cleaning and preparing data
- Feature Engineering: Creating new features and handling imbalanced data with SMOTE
- Modeling: Training and evaluating Random Forest, Decision Tree, XGBoost models
- Explainability: SHAP module for interpreting model predictions
- Backend: Flask web server managing API and model predictions
- Frontend: HTML/CSS/JavaScript interface for user interactions

---

## Technologies Used

- Programming Languages: Python, HTML, CSS, JavaScript
- Machine Learning: scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP
- Backend Framework: Flask
- Visualization: Matplotlib, Seaborn
- Version Control: Git and GitHub

---

## Dependencies

- pandas
- numpy
- scikit-learn
- xgboost
- imbalanced-learn
- shap
- flask
- matplotlib
- seaborn
- pickle-mixin (for serialization)

(Ensure to install these via `pip install -r requirements.txt`)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Authors

- Jeevanraj L — [GitHub](https://github.com/Jeevan-cyber-ai) — jeevaneniyavan@gmail.com  
- Anisha A K — [GitHub](https://github.com/Anisha452531) — anishaak06@gmail.com  
- Madhanalekha — [GitHub](https://github.com/Madhanalekha) — madhanalekha203@gmail.com  
- Thangappan S — [GitHub](https://github.com/thangappans) — sthangappan77@gmail.com  
- Gowsalya — [GitHub](https://github.com/sathya292006) — sathyamozhi292006@gmail.com  


