from flask import Flask, render_template, request
from ShapModule import shap_explanation
from NewChurnPrediction import predict_churn_for_new_customer

app = Flask(_name_)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Case 1: Predict by existing customer index
        if "customer_index" in request.form and request.form["customer_index"].strip() != "":
            index = int(request.form["customer_index"])
            file_path = "Churn_Modelling.csv"
            prediction, probability, shap_list = shap_explanation(file_path, index)

        # Case 2: Predict for a new customer
        else:
            input_data = {
                'CreditScore': int(request.form["CreditScore"]),
                'Geography': request.form["Geography"],
                'Gender': request.form["Gender"],
                'Age': int(request.form["Age"]),
                'Tenure': int(request.form["Tenure"]),
                'Balance': float(request.form["Balance"]),
                'NumOfProducts': int(request.form["NumOfProducts"]),
                'HasCrCard': int(request.form["HasCrCard"]),
                'EstimatedSalary': float(request.form["EstimatedSalary"])
            }

            prediction, probability, shap_list = predict_churn_for_new_customer(input_data)

        # Render result page
        return render_template("result.html", prediction=prediction, probability=f"{probability:.2f}", shap_values=shap_list)

    except Exception as e:
        # Error handling: Show error on the same page
        return f"<h3 style='color:red;'>Error occur: {str(e)}</h3><a href='/'>Go back</a>"

if _name_ == "_main_":
    app.run(debug=True)