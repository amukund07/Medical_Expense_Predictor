from flask import Flask, render_template, request
import LR

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        age = request.form["age"]
        bmi = request.form["bmi"]
        children = request.form["children"]
        sex = request.form["sex"]
        smoker = request.form["smoker"]
        region = request.form["area"]

        prediction = LR.predict_charge(age, bmi, children, sex, smoker, region)

        return render_template("index.html", prediction=prediction)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
