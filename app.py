from flask import Flask, render_template, request
import numpy as np
import pickle
import logging
import pandas as pd

# Logging
logging.basicConfig(level=logging.DEBUG)

# Initialize app
app = Flask(__name__)

# Load models and scalers
try:
    # سكر
    diabetes_data = pickle.load(open('models/Diabetes_model_KNeighborsClassifier.pkl', 'rb'))
    diabetes_model = diabetes_data['model']
    diabetes_scaler = diabetes_data['scaler']

    # توحد
    autism_model = pickle.load(open('models/Autism.pkl', 'rb'))
    autism_scaler = pickle.load(open('models/Scaler_autism.pkl', 'rb'))

    # انيميا
    anemia_data = pickle.load(open('models/Animia_model.pkl', 'rb'))
    anemia_model = anemia_data['model']
    anemia_scaler = anemia_data['scaler']

    # كورونا
    corona_data = pickle.load(open('models/Covid_2(final_logistic).pkl', 'rb'))
    corona_model = corona_data['model']
    corona_scaler = corona_data['scaler']

    # قلب
    heart_diseaese_data = pickle.load(open('models/heart.pkl', 'rb'))
    heart_model = heart_diseaese_data['model']
    heart_scaler = heart_diseaese_data['scaler']
    heart_columns_to_scale = heart_diseaese_data['columns_to_scale']

    #كبد
    liver_data=pickle.load(open('models/Liver_Patients.pkl', 'rb'))
    liver_model=liver_data['model']
    liver_scaler=liver_data['scaler']


except Exception as e:
    logging.error("Error loading models or scalers: %s", e)
    raise

# Feature columns for Autism
autism_features = [
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
    'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age', 'gender',
    'ethnicity', 'jundice', 'austim'
]

@app.route('/')
def home():
    return render_template('home.html')

# سكر
@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        try:
            features = [
                float(request.form['Pregnancies']),
                float(request.form['Glucose']),
                float(request.form['BloodPressure']),
                float(request.form['SkinThickness']),
                float(request.form['Insulin']),
                float(request.form['BMI']),
                float(request.form['DiabetesPedigreeFunction']),
                float(request.form['Age'])
            ]

            input_data = np.array([features])
            input_scaled = diabetes_scaler.transform(input_data)
            prediction = diabetes_model.predict(input_scaled)

            result = "🔴 المريض مُحتمل أن يكون مصابًا بالسكري" if prediction[0] == 1 else "🟢 المريض غير مصاب بالسكري"
            return render_template('final_result.html', result=result)
        except ValueError:
            return render_template('final_result.html', result="⚠ من فضلك تأكد من إدخال أرقام صحيحة في كل الخانات.")
        except Exception as e:
            logging.error("Error during diabetes prediction: %s", e)
            return render_template('final_result.html', result="⚠ حدث خطأ أثناء المعالجة، يرجى المحاولة لاحقًا.")
    return render_template('diabetes_form.html')

# انيميا
@app.route('/animia', methods=['GET', 'POST'])
def animia():
    if request.method == 'POST':
        try:
            features = [
                float(request.form['Hemoglobin']),
                float(request.form['MCH']),
                float(request.form['MCHC']),
                float(request.form['MCV']),
            ]
            input_data = np.array([features])
            input_scaled = anemia_scaler.transform(input_data)
            prediction = anemia_model.predict(input_scaled)
            result = "🔴 المريض مُحتمل أن يكون مصابًا بالأنيميا" if prediction[0] == 1 else "🟢 المريض غير مصاب بالأنيميا"
            return render_template('final_result.html', result=result)
        except ValueError:
            return render_template('final_result.html', result="⚠ من فضلك تأكد من إدخال أرقام صحيحة في كل الخانات.")
        except Exception as e:
            logging.error("Error during anemia prediction: %s", e)
            return render_template('final_result.html', result="⚠ حدث خطأ أثناء المعالجة، يرجى المحاولة لاحقًا.")
    return render_template('animia_form.html')

# كورونا
@app.route('/corona', methods=['GET', 'POST'])
def corona():
    if request.method == 'POST':
        try:
            features = [
                int(request.form['usmer']),
                int(request.form['medical_unit']),
                int(request.form['sex']),
                int(request.form['patient_type']),
                float(request.form['pneumonia']),
                float(request.form['age']),
                float(request.form['pregnant']),
                float(request.form['diabetes']),
                float(request.form['copd']),
                float(request.form['asthma']),
                float(request.form['inmsupr']),
                float(request.form['hipertension']),
                float(request.form['other_disease']),
                float(request.form['cardiovascular']),
                float(request.form['obesity']),
                float(request.form['renal_chronic']),
                float(request.form['tobacco']),
                int(request.form['clasiffication_final'])
            ]

            columns = [
                'usmer', 'medical_unit', 'sex', 'patient_type', 'pneumonia', 'age', 
                'pregnant', 'diabetes', 'copd', 'asthma', 'inmsupr', 'hipertension', 
                'other_disease', 'cardiovascular', 'obesity', 'renal_chronic', 'tobacco', 'clasiffication_final'
            ]

            input_data = pd.DataFrame([features], columns=columns)
            input_data['age'] = corona_scaler.transform(input_data[['age']])
            prediction = corona_model.predict(input_data)

            result = "🔴 المريض مُحتمل أن يكون في حالة خطيرة بسبب الكورونا" if prediction[0] == 1 else "🟢 المريض في حالة مستقرة"
            return render_template('final_result.html', result=result)
        
        except ValueError:
            return render_template('final_result.html', result="⚠ من فضلك تأكد من إدخال أرقام صحيحة في كل الخانات.")
        except Exception as e:
            logging.error("Error during corona prediction: %s", e)
            return render_template('final_result.html', result="⚠ حدث خطأ أثناء المعالجة، يرجى المحاولة لاحقًا.")
    
    return render_template('covid_form.html')

# قلب

@app.route('/heart', methods=['GET', 'POST'])
def heart():
    if request.method == 'POST':
        try:
            data = {
                'age': float(request.form['age']),
                'resting_blood_pressure': float(request.form['resting_blood_pressure']),
                'cholesterol': float(request.form['cholesterol']),
                'fasting_blood_sugar': float(request.form['fasting_blood_sugar']),
                'max_heart_rate_achieved': float(request.form['max_heart_rate_achieved']),
                'exercise_induced_angina': float(request.form['exercise_induced_angina']),
                'st_depression': float(request.form['st_depression']),
                'sex_male': 1 if request.form.get('sex') == 'male' else 0,

                'chest_pain_type_atypical angina': 1 if request.form['chest_pain_type'] == 'atypical angina' else 0,
                'chest_pain_type_non-anginal pain': 1 if request.form['chest_pain_type'] == 'non-anginal pain' else 0,
                'chest_pain_type_typical angina': 1 if request.form['chest_pain_type'] == 'typical angina' else 0,

                'rest_ecg_left ventricular hypertrophy': 1 if request.form['rest_ecg'] == 'left ventricular hypertrophy' else 0,
                'rest_ecg_normal': 1 if request.form['rest_ecg'] == 'normal' else 0,

                'st_slope_flat': 1 if request.form['st_slope'] == 'flat' else 0,
                'st_slope_normal': 1 if request.form['st_slope'] == 'normal' else 0,
                'st_slope_upsloping': 1 if request.form['st_slope'] == 'upsloping' else 0
            }

            features = pd.DataFrame([data], columns=[
                'age', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar',
                'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression',
                'sex_male', 'chest_pain_type_atypical angina', 'chest_pain_type_non-anginal pain',
                'chest_pain_type_typical angina', 'rest_ecg_left ventricular hypertrophy',
                'rest_ecg_normal', 'st_slope_flat', 'st_slope_normal', 'st_slope_upsloping'
            ])

            features[heart_columns_to_scale] = heart_scaler.transform(features[heart_columns_to_scale])
            prediction = heart_model.predict(features)[0]
            proba = heart_model.predict_proba(features)[0]

            result = "🔴 النتيجة إيجابية لمرض القلب" if prediction == 1 else "🟢 النتيجة سلبية لمرض القلب"
            confidence = round(max(proba) * 100, 2)

            return render_template('final_result.html', result=result, confidence=confidence)

        except Exception as e:
            logging.error("Heart disease prediction error: %s", e)
            return render_template('final_result.html', result="⚠ حدث خطأ أثناء المعالجة.", confidence=0)

    return render_template('heart_form.html')


#كبد
# liver disease
@app.route('/liver', methods=['GET', 'POST'])
def liver():
    if request.method == 'POST':
        try:
            gender = request.form['Gender of the patient']
            gender_numeric = 1 if gender == 'Male' else 0

            features = [
                float(request.form['Age of the patient']),
                gender_numeric,
                float(request.form['Total Bilirubin']),
                float(request.form['Direct Bilirubin']),
                float(request.form['Alk Phos Alkaline Phosphatase']),
                float(request.form['Sgpt Alanine Aminotransferase']),
                float(request.form['Sgot Aspartate Aminotransferase']),
                float(request.form['Total Proteins']),
                float(request.form['ALB Albumin']),
                float(request.form['A/G Ratio Albumin and Globulin Ratio'])
            ]

            input_data = np.array([features])
            input_scaled = liver_scaler.transform(input_data)
            prediction = liver_model.predict(input_scaled)

            result = "🔴 المريض مُحتمل أن يكون مصابًا بمرض في الكبد" if prediction[0] == 1 else "🟢 المريض غير مصاب بمرض في الكبد"
            return render_template('final_result.html', result=result)
        except ValueError:
            return render_template('final_result.html', result="⚠ من فضلك تأكد من إدخال أرقام صحيحة في كل الخانات.")
        except Exception as e:
            logging.error("Error during liver prediction: %s", e)
            return render_template('final_result.html', result="⚠ حدث خطأ أثناء المعالجة، يرجى المحاولة لاحقًا.")
    return render_template('Liver_Patients_form.html')  # لو حبيت تخليها صفحة منفصلة


# توحد
@app.route('/predict', methods=['GET', 'POST'])
def autism():
    if request.method == 'POST':
        try:
            features = {
                'A1_Score': int(request.form['A1_Score']),
                'A2_Score': int(request.form['A2_Score']),
                'A3_Score': int(request.form['A3_Score']),
                'A4_Score': int(request.form['A4_Score']),
                'A5_Score': int(request.form['A5_Score']),
                'A6_Score': int(request.form['A6_Score']),
                'A7_Score': int(request.form['A7_Score']),
                'A8_Score': int(request.form['A8_Score']),
                'A9_Score': int(request.form['A9_Score']),
                'A10_Score': int(request.form['A10_Score']),
                'age': float(request.form['age']),
                'gender': int(request.form['gender']),
                'ethnicity': int(request.form['ethnicity']),
                'jundice': int(request.form['jundice']),
                'austim': int(request.form['austim']),
            }

            input_df = pd.DataFrame([features])[autism_features]
            scaled_input = autism_scaler.transform(input_df)
            prediction = autism_model.predict(scaled_input)[0]
            proba = autism_model.predict_proba(scaled_input)[0]

            result = "🔴 النتيجة إيجابية للتوحد" if prediction == 1 else "🟢 النتيجة سلبية للتوحد"
            confidence = round(max(proba) * 100, 2)

            return render_template('final_result.html', result=result, confidence=confidence)
        except Exception as e:
            logging.error("Autism error: %s", e)
            return render_template('final_result.html', result="⚠ خطأ أثناء المعالجة.", confidence=0)
    return render_template('autism_form.html')

if __name__ == '__main__':
    app.run(debug=True)


