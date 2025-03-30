from flask import Flask, request, render_template, redirect, url_for, session, flash
import numpy as np
import pandas as pd
import joblib
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
import matplotlib.pyplot as plt
import io
import base64
from flask_apscheduler import APScheduler
from datetime import datetime, timedelta

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Flask-Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Gmail SMTP Server
app.config['MAIL_PORT'] = 587  # TLS Port
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'poojaryharshith2004@gmail.com'
app.config['MAIL_PASSWORD'] = ''
app.config['MAIL_DEFAULT_SENDER'] = 'poojaryharshith2004@gmail.com'

# Initialize Flask-Mail
mail = Mail(app)
#Scheduling the Medication
# Initialize APScheduler
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()


def send_medication_reminders():
    with app.app_context():  # Ensure Flask context is available
        now = datetime.now()
        one_hour_later = now + timedelta(hours=1)

        conn = get_db_connection()
        cursor = conn.cursor()

        # Fetch medications scheduled within the next hour
        cursor.execute("""
            SELECT med_id, user_id, medicine_name, date, time 
            FROM medications 
            WHERE status = 'Scheduled' 
            AND date = ? 
            AND time BETWEEN ? AND ?
        """, (now.strftime('%Y-%m-%d'), now.strftime('%H:%M'), one_hour_later.strftime('%H:%M')))

        reminders = cursor.fetchall()

        # Fetch user emails
        user_emails = {}
        user_ids = list(set(reminder['user_id'] for reminder in reminders))

        if user_ids:
            query = f"SELECT id, email FROM users WHERE id IN ({','.join('?' * len(user_ids))})"
            cursor.execute(query, user_ids)
            user_emails = {row['id']: row['email'] for row in cursor.fetchall()}

        # Send emails and update the status
        for reminder in reminders:
            med_id = reminder['med_id']
            user_id = reminder['user_id']
            medicine_name = reminder['medicine_name']
            scheduled_time = f"{reminder['date']} {reminder['time']}"

            if user_id in user_emails:
                user_email = user_emails[user_id]
                msg = Message("Medication Reminder", recipients=[user_email])
                msg.body = f"Hello,\n\nIt's time to take your medication: {medicine_name} at {scheduled_time}.\n\nRegards,\nMedical Expert System"

                try:
                    mail.send(msg)
                    print(f"📧 Email sent successfully to {user_email} for {medicine_name}.")

                    # Update medication status to "Sent"
                    cursor.execute("UPDATE medications SET status = 'Sent' WHERE med_id = ?", (med_id,))
                    conn.commit()
                    print(f"✅ Status updated to 'Sent' for {medicine_name}.")

                except Exception as e:
                    print(f"⚠️ Failed to send email to {user_email}: {str(e)}")

        conn.close()


# Schedule reminder job to run every 30 minutes (instead of 1 hour for better accuracy)
scheduler.add_job(id='medication_reminders', func=send_medication_reminders, trigger='interval', minutes=1)

# Load dataset files
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets.csv")

# Load trained ML model
svc = joblib.load(open("models/svc.pkl", "rb"))

# Database connection function
def get_db_connection():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

# Helper function to retrieve disease-related data
def helper(dis):
    # Retrieve and format disease description
    desc = description[description['Disease'] == dis]['Description'].astype(str).tolist()
    desc = " ".join(desc) if desc else "No description available."

    # Retrieve and properly format precautions
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    # Retrieve and properly format medications
    med = medications[medications['Disease'] == dis]['Medication'].astype(str).tolist()
    med = med if med else ["No medications available"]

    # Retrieve and properly format diets
    die = diets[diets['Disease'] == dis]['Diet'].astype(str).tolist()
    die = die if die else ["No diet available"]

    # Retrieve and properly format workout suggestions
    wrkout = workout[workout['disease'] == dis]['workout'].astype(str).tolist()
    wrkout = wrkout if wrkout else ["No workout suggestions"]

    return desc, pre, med, die, wrkout


symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# ------------------- Flask Routes -------------------

# Home Route (Redirects to Login if not logged in)
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

# Dashboard Route (Requires Login)
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('You need to log in first.', 'warning')
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch user's appointments
    cursor.execute("SELECT * FROM appointments WHERE user_id = ?", (session['user_id'],))
    appointments = cursor.fetchall()

    # Fetch health data with default values if None
    cursor.execute("SELECT bmi, blood_pressure, glucose_level FROM users WHERE id = ?", (session['user_id'],))
    health_data = cursor.fetchone()

    if health_data is None:
        health_data = {'bmi': 'N/A', 'blood_pressure': '0/0', 'glucose_level': 'N/A'}
    else:
        health_data = dict(health_data)
        health_data['blood_pressure'] = health_data['blood_pressure'] or '0/0'

    # Fetch updated medication reminders
    cursor.execute("""
        SELECT * FROM medications 
        WHERE user_id = ? 
        ORDER BY date ASC, time ASC
    """, (session['user_id'],))
    medications = cursor.fetchall()

    conn.close()

    return render_template(
        'dashboard.html',
        username=session['username'],
        appointments=appointments,
        health_data=health_data,  # Ensuring non-None health data
        medications=medications  # Now passing updated medications
    )


# History Route (Requires Login)
@app.route('/history')
def history():
    if 'user_id' not in session:
        flash('Please log in to access your medical history.', 'warning')
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT symptoms, predicted_disease, description, precautions, medications, diet, workout, diagnosis_date
           FROM history WHERE user_id = ? ORDER BY diagnosis_date DESC""",
        (session['user_id'],)
    )
    history_records = cursor.fetchall()
    conn.close()

    return render_template('history.html', history=history_records)


# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password. Please try again.', 'danger')

    return render_template('login.html')

# Register Route
import random


@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if email already exists
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        existing_user = cursor.fetchone()
        conn.close()

        if existing_user:
            flash('Email already exists. Please log in.', 'danger')
            return redirect(url_for('login'))

        # ✅ Generate a 6-digit OTP
        otp = str(random.randint(100000, 999999))

        # ✅ Store details temporarily in session
        session['otp'] = otp
        session['username'] = username
        session['email'] = email
        session['password'] = password  # Store hashed password temporarily

        # ✅ Send OTP email
        try:
            msg = Message("Email Verification - OTP", recipients=[email])
            msg.body = f"Hello {username},\n\nYour OTP for email verification is: {otp}\n\nRegards,\nMedical Expert System"
            mail.send(msg)
            flash('OTP sent to your email. Please verify.', 'info')
            return redirect(url_for('verify_otp'))
        except Exception as e:
            flash(f"Error sending OTP: {str(e)}", 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if 'otp' not in session:
        flash('Session expired. Please register again.', 'danger')
        return redirect(url_for('register'))

    if request.method == 'POST':
        entered_otp = request.form['otp']

        if entered_otp == session.get('otp'):
            # ✅ OTP Matched - Register the User
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (session['username'], session['email'], session['password'])
            )
            conn.commit()
            conn.close()

            flash('Account created successfully! You can now log in.', 'success')

            # ✅ Clear OTP and session data after successful registration
            session.pop('otp', None)
            session.pop('username', None)
            session.pop('email', None)
            session.pop('password', None)

            return redirect(url_for('login'))
        else:
            flash('Invalid OTP. Please try again.', 'danger')

    return render_template('verify_otp.html')


# Get Diagnosis Route
@app.route('/get_diagnosis')
def get_diagnosis():
    if 'user_id' not in session:
        flash('You need to log in first.', 'warning')
        return redirect(url_for('login'))
    return render_template('index.html')

# Predict Route (As provided)
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        print("User Symptoms:", symptoms)

        if not symptoms or symptoms == "Symptoms":
            message = "Please enter valid symptoms."
            return render_template('index.html', message=message)

        # Convert user input into a list of symptoms
        user_symptoms = [s.strip() for s in symptoms.split(',')]

        # Get predicted disease
        predicted_disease = get_predicted_value(user_symptoms)
        dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

        # Store the full details in history if user is logged in
        if 'user_id' in session:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Save the symptoms, predicted disease, and related details in the database
            cursor.execute(
                """INSERT INTO history (user_id, symptoms, predicted_disease, description, precautions, medications, diet, workout)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session['user_id'], ', '.join(user_symptoms), predicted_disease, dis_des, ', '.join(precautions[0]),
                 ', '.join(medications), ', '.join(rec_diet), ', '.join(workout))
            )
            conn.commit()
            conn.close()

        # Handle empty precautions list
        my_precautions = []
        if precautions and len(precautions) > 0:
            for i in precautions[0]:
                my_precautions.append(i)
        else:
            my_precautions.append("No precautions available")

        return render_template(
            'index.html',
            predicted_disease=predicted_disease,
            dis_des=dis_des,
            my_precautions=my_precautions,
            medications=medications,
            my_diet=rec_diet,
            workout=workout
        )

    return render_template('index.html')


# Logout Route
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# About Page
@app.route('/about')
def about():
    return render_template('about.html')

# Contact Route
@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        message = request.form["message"]

        # Save the message in the database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO contact_messages (name, email, message) VALUES (?, ?, ?)",
                       (name, email, message))
        conn.commit()
        conn.close()

        # Optional: Send a confirmation email
        try:
            msg = Message("Contact Form Received", recipients=[email])
            msg.body = f"Hello {name},\n\nThank you for reaching out to us!\n\nWe have received your message and will get back to you soon.\n\nYour Message:\n{message}\n\nBest Regards,\nHealth Care Center"
            mail.send(msg)
        except Exception as e:
            flash(f"Message saved, but email not sent: {str(e)}", "warning")

        flash("Your message has been sent successfully!", "success")
        return redirect(url_for("contact"))

    return render_template("contact.html")

# Developer Page
@app.route('/developer')
def developer():
    return render_template('developer.html')

# Blog Page
@app.route('/blog')
def blog():
    return render_template('blog.html')

# Appointment Routes
@app.route('/add_appointment', methods=['GET', 'POST'])
def add_appointment():
    if 'user_id' not in session:
        flash('Please log in first', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        doctor_name = request.form['doctor_name']
        doctor_email = request.form['doctor_email']
        appointment_date = request.form['appointment_date']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO appointments (user_id, doctor_name, doctor_email, appointment_date, status) VALUES (?, ?, ?, ?, ?)",
            (session['user_id'], doctor_name, doctor_email, appointment_date, 'Scheduled')
        )
        conn.commit()
        appointment_id = cursor.lastrowid  # Get the inserted appointment ID
        conn.close()

        # Generate confirmation and rejection links
        confirm_link = url_for('confirm_appointment', appointment_id=appointment_id, status='Confirmed', _external=True)
        reject_link = url_for('confirm_appointment', appointment_id=appointment_id, status='Rejected', _external=True)

        # Send Email Notification to Doctor
        try:
            msg = Message("New Appointment Scheduled", recipients=[doctor_email])
            msg.body = f"""
            Hello Dr. {doctor_name},

            A new appointment has been scheduled with you.

            Patient Name: {session['username']}
            Appointment Date: {appointment_date}

            Please confirm or reject the appointment by clicking below:

            ✅ Confirm: {confirm_link}
            ❌ Reject: {reject_link}

            Regards,
            Medical Expert System
            """
            mail.send(msg)
            flash('Appointment successfully scheduled! An email has been sent to the doctor.', 'success')

        except Exception as e:
            flash(f'Appointment scheduled, but email could not be sent: {str(e)}', 'warning')

        return redirect(url_for('dashboard'))

    return render_template('add_appointment.html')


@app.route('/confirm_appointment/<int:appointment_id>/<status>')
def confirm_appointment(appointment_id, status):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Update the status of the appointment
    cursor.execute("UPDATE appointments SET status = ? WHERE appointment_id = ?", (status, appointment_id))
    conn.commit()
    conn.close()

    return f"Appointment {status}! Thank you for your response."


@app.route('/appointments')
def appointments():
    if 'user_id' not in session:
        flash('Please log in first', 'warning')
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM appointments WHERE user_id = ?", (session['user_id'],))
    appointments = cursor.fetchall()
    conn.close()

    return render_template('appointments.html', appointments=appointments)

@app.route('/reschedule_appointment/<int:appointment_id>', methods=['GET', 'POST'])
def reschedule_appointment(appointment_id):
    if 'user_id' not in session:
        flash('Please log in first', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        new_date = request.form['new_appointment_date']

        conn = get_db_connection()
        cursor = conn.cursor()

        # First, retrieve doctor email before closing the connection
        cursor.execute("SELECT doctor_email, doctor_name FROM appointments WHERE appointment_id = ?", (appointment_id,))
        appointment = cursor.fetchone()

        if not appointment:
            conn.close()  # Ensure connection is closed
            flash("Appointment not found.", "danger")
            return redirect(url_for('appointments'))

        doctor_email = appointment['doctor_email']
        doctor_name = appointment['doctor_name']

        # Now update the appointment status
        cursor.execute("UPDATE appointments SET appointment_date = ?, status = 'Rescheduled' WHERE appointment_id = ?",
                       (new_date, appointment_id))
        conn.commit()
        conn.close()  # Close the connection after all operations are done

        # Send email notification to doctor
        try:
            msg = Message("Appointment Rescheduled", recipients=[doctor_email])
            msg.body = f"""
            Hello Dr. {doctor_name},

            The following appointment has been **rescheduled**:

            Patient Name: {session['username']}
            New Appointment Date: {new_date}

            Please update your schedule accordingly.

            Regards,
            Medical Expert System
            """
            mail.send(msg)
            flash('Appointment successfully rescheduled. The doctor has been notified.', 'success')
        except Exception as e:
            flash(f'Appointment rescheduled, but email could not be sent: {str(e)}', 'warning')

        return redirect(url_for('appointments'))

    return render_template('reschedule_appointment.html', appointment_id=appointment_id)

@app.route('/cancel_appointment/<int:appointment_id>', methods=['GET', 'POST'])
def cancel_appointment(appointment_id):
    if 'user_id' not in session:
        flash('Please log in first', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        cancellation_reason = request.form['cancellation_reason']

        conn = get_db_connection()
        cursor = conn.cursor()

        # First, retrieve doctor email before closing the connection
        cursor.execute("SELECT doctor_email, doctor_name FROM appointments WHERE appointment_id = ?", (appointment_id,))
        appointment = cursor.fetchone()

        if not appointment:
            conn.close()  # Ensure connection is closed
            flash("Appointment not found.", "danger")
            return redirect(url_for('appointments'))

        doctor_email = appointment['doctor_email']
        doctor_name = appointment['doctor_name']

        # Now update the appointment status
        cursor.execute("UPDATE appointments SET status = 'Cancelled', cancellation_reason = ? WHERE appointment_id = ?",
                       (cancellation_reason, appointment_id))
        conn.commit()
        conn.close()  # Close the connection after all operations are done

        # Send email notification to doctor
        try:
            msg = Message("Appointment Cancelled", recipients=[doctor_email])
            msg.body = f"""
            Hello Dr. {doctor_name},

            The following appointment has been **cancelled**:

            Patient Name: {session['username']}
            Reason: {cancellation_reason}

            Please remove this appointment from your schedule.

            Regards,
            Medical Expert System
            """
            mail.send(msg)
            flash('Appointment successfully cancelled. The doctor has been notified.', 'success')
        except Exception as e:
            flash(f'Appointment cancelled, but email could not be sent: {str(e)}', 'warning')

        return redirect(url_for('appointments'))

    return render_template('cancel_appointment.html', appointment_id=appointment_id)

@app.route('/update_health', methods=['GET', 'POST'])
def update_health():
    if 'user_id' not in session:
        flash('Please log in first.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        bmi = request.form['bmi']
        blood_pressure = request.form['blood_pressure']
        glucose_level = request.form['glucose_level']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET bmi = ?, blood_pressure = ?, glucose_level = ? WHERE id = ?",
            (bmi, blood_pressure, glucose_level, session['user_id'])
        )
        conn.commit()
        conn.close()

        flash('Health data updated successfully!', 'success')
        return redirect(url_for('dashboard'))

    return render_template('update_health.html')

@app.route('/health_chart')
def health_chart():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT bmi, glucose_level FROM users WHERE id = ?", (session['user_id'],))
    health_data = cursor.fetchone()
    conn.close()

    bmi, glucose = health_data['bmi'], health_data['glucose_level']

    # Create plot
    plt.figure(figsize=(5, 3))
    labels = ["BMI", "Glucose Level"]
    values = [bmi, glucose]
    plt.bar(labels, values, color=['blue', 'red'])
    plt.title("Health Stats")

    # Save plot to a BytesIO buffer
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()

    return f'<img src="data:image/png;base64,{chart_url}" />'

#Add Medication
@app.route('/add_medication', methods=['GET', 'POST'])
def add_medication():
    if 'user_id' not in session:
        flash('Please log in first.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        medicine_name = request.form['medicine_name']
        date = request.form['date']
        time = request.form['time']
        status = 'Scheduled'  # Default status

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO medications (user_id, medicine_name, date, time, status) VALUES (?, ?, ?, ?, ?)",
            (session['user_id'], medicine_name, date, time, status)
        )
        conn.commit()
        conn.close()

        flash('Medication scheduled successfully!', 'success')
        return redirect(url_for('dashboard'))

    return render_template('add_medication.html')


#Show Medication
@app.route('/medications')
def view_medications():  # Changed function name
    if 'user_id' not in session:
        flash('Please log in first.', 'warning')
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM medications WHERE user_id = ?", (session['user_id'],))
    meds = cursor.fetchall()  # Renamed variable to avoid conflict
    conn.close()

    return render_template('medications.html', medications=meds)  # Ensure correct reference

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
