from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        data = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]

        input_data = np.array(data).reshape(1, -1)

        # Predict the class and probability
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]  # probability of being diabetic

        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        probability_percent = round(probability * 100, 2)

        # Determine severity stage based on probability
        if probability < 0.3:
            severity = "Low Risk"
            severity_class = "low-risk"
        elif probability < 0.6:
            severity = "Moderate Risk"
            severity_class = "moderate-risk"
        elif probability < 0.85:
            severity = "High Risk"
            severity_class = "high-risk"
        else:
            severity = "Very High Risk"
            severity_class = "very-high-risk"

    except Exception as e:
        result = f"Error: {str(e)}"
        severity = ""
        probability_percent = ""
        severity_class = ""

    return render_template(
        'index.html',
        result=result,
        severity=severity,
        probability=probability_percent,
        severity_class=severity_class
    )

if __name__ == '__main__':
    app.run(debug=True)
