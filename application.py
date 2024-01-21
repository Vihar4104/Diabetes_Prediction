import pickle
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application   

Log_reg = pickle.load(open('model/Prediction.pkl', 'rb'))
scaler_model = pickle.load(open('model/StandardScaler.pkl', 'rb'))

## Route for Home Page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    result = ""
    if request.method == 'POST':
        Pregnancies = float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))
        

        # Don't reassign scaler_model here
        new_data_scaled = scaler_model.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        predict = Log_reg.predict(new_data_scaled)

        if predict[0] ==1 :
            result = "Diebetics"

        else:
            result = "No Diebetics"

        return render_template('home.html', result=result)

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
