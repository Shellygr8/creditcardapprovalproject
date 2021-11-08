#Importing the liabraries
import pickle
from flask import Flask, render_template, request

#Global Variables
app = Flask(__name__)
loadedModel = pickle.load(open('Model.pkl', 'rb'))

#Routes
@app.route('/')
def home():
    return render_template('form.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    monthly_balance = request.form['monthly balance']
    total_income = request.form['total income']
    gender = request.form['gender']
    housing_type = request.form['housing type']
    family_status = request.form['family status']
    income_type = request.form['income type']

    prediction = loadedModel.predict([[monthly_balance, gender, housing_type, family_status, income_type, total_income]])[0]

    if prediction == 0:
        prediction = "Bad Candidate For Credit Card"
    else:
        prediction = "Bad Candidate For Credit Card"

    return render_template('form.html', output = prediction)

#Main functions
if __name__ == '__main__':
    app.run(debug=True)