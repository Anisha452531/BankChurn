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

## ⚙ Installation Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/Anisha452531/BankChurn.git
cd BankChurn
