from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        glucose = float(request.form["Glucose"])
        result = "Diabetic" if glucose > 125 else "Not Diabetic"
    except Exception as e:
        result = f"Error: {str(e)}"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
