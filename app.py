#Importing the liabraries
import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd

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
    family_members = request.form['family members']
    own_car = request.form['own car']
    income_type = request.form['income type']

#     Index(['MONTHS_BALANCE', 'CODE_GENDER', 'NAME_HOUSING_TYPE',
#        'NAME_FAMILY_STATUS', 'CNT_FAM_MEMBERS', 'NAME_INCOME_TYPE',
#        'FLAG_OWN_CAR', 'AMT_INCOME_TOTAL'],
#       dtype='object')

    testArr = np.array([[monthly_balance, gender, housing_type, family_status, family_members, income_type, own_car, total_income]])
    testCol = loadedModel.get_booster().feature_names
    testDf = pd.DataFrame(data=testArr, columns=testCol)

    prediction =  loadedModel.predict(testDf)[0]

    if prediction == 0:
        prediction = "Bad Candidate For Credit Card"
    else:
        prediction = "Good Candidate For Credit Card"

    return render_template('form.html', output = prediction)

#Main functions
if __name__ == '__main__':
    app.run(debug=True)
