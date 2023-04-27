from flask import Flask, render_template, request
from find_top_hubs.find_top_hubs import find_top_hubs

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('3-start-here.html')

@app.route('/result', methods=['POST'])
def result():
    clinical_trial_type = request.form['clinical_trial_type']
    disease_focus = request.form['disease_focus']
    age = int(request.form['age'])
    gender = request.form['gender']
    race_ethnicity = request.form['race_ethnicity']
    education_level = request.form['education_level']
    income = int(request.form['income'])

    top_hubs = find_top_hubs(age, gender, race_ethnicity, education_level, income)
    print(top_hubs)

    return render_template('4-result.html', top_hubs=top_hubs)

if __name__ == '__main__':
    app.run(debug=True)
