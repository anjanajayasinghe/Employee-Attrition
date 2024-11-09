# app.py

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained AdaBoost model from the pickle file
model_path = "models\picklefile.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')  # Render your HTML form

@app.route('/submit', methods=['POST'])
def submit():
    
    # Collect data from the form
    age = request.form.get('Age')
    gender = request.form.get('Gender')
    #job_role = request.form.get('Job Role')
    overtime = request.form.get('Overtime')
    marital_status = request.form.get('Marital Status')
    remote_work = request.form.get('Remote Work')
    leadership_opportunities = request.form.get('Leadership Opportunities')
    innovation_opportunities = request.form.get('Innovation Opportunities')
    work_life_balance = request.form.get('Work-Life Balance')
    
    job_satisfaction = request.form.get('Job Satisfaction')
    performance_rating = request.form.get('Performance Rating')
    education_level = request.form.get('Education Level')
    job_level = request.form.get('Job Level')
    company_size = request.form.get('Company Size')
    company_reputation = request.form.get('Company Reputation')
    employee_recognition = request.form.get('Employee Recognition')
    
    years_at_company = request.form.get('Years at Company')
    monthly_income = request.form.get('Monthly Income')
    number_of_promotions = request.form.get('number_of_promotions')
    distance_from_home = request.form.get('Distance from Home')
    number_of_dependents = request.form.get('number_of_dependents')
    company_tenure = request.form.get('Company Tenure')
    
    feedback = request.form.get('feedback', '')  # Optional field

    # Prepare the input data for prediction (convert categorical variables as needed)
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        #'Job Role': [job_role],
        'Overtime': [overtime],
        'Marital Status': [marital_status],
        'Remote Work': [remote_work],
        'Leadership Opportunities': [leadership_opportunities],
        'Innovation Opportunities': [innovation_opportunities],
        'Work-Life Balance': [work_life_balance],
        'Job Satisfaction': [job_satisfaction],
        'Performance Rating': [performance_rating],
        'Education Level': [education_level],
        'Job Level': [job_level],
        'Company Size': [company_size],
        'Company Reputation': [company_reputation],
        'Employee Recognition': [employee_recognition],
        'Years at Company': [years_at_company],
        'Monthly Income': [monthly_income],
        'Number of Promotions': [number_of_promotions],
        'Distance from Home': [distance_from_home],
        'Number of Dependents': [number_of_dependents],
        'Company Tenure': [company_tenure]
    })

    # Make prediction using the loaded model
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Get the probability of the employee leaving (class 1)
    leave_probability = prediction_proba[0][1]  # Probability of class 1 (LEFT)

    # Redirect or render a template with the prediction result
    return render_template('result1.html', prediction=prediction[0], leave_probability=leave_probability)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)