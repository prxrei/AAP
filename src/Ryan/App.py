import joblib
from flask import Flask, render_template, request, jsonify, redirect, url_for

app = Flask(__name__)


model = joblib.load("LogRegModel.pkl")


features = [
    "Attendance", "Age",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", 
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular Units 1st sem (grade)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (without evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular Units 2nd sem (grade)"
]



@app.route("/")
def home():
    return redirect(url_for("Predictresults"))

@app.route("/Predictresults", methods=["GET", "POST"])
def Predictresults():
    if request.method == "GET":
        return render_template("Predictresults.html")
    else:
        
        input_data = request.form

        
        data = {
            feature: int(input_data.get(feature, 0)) for feature in features
        }

        
        prediction = model.predict([list(data.values())])

       
        prediction_result = "Failing" if prediction[0] == 0 else "Passing"  

        
        return redirect(url_for("result", result=prediction_result))


@app.route("/Prediction", methods=["GET"])
def result():
    
    result = request.args.get("result")
    
    
    return render_template("Prediction.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
