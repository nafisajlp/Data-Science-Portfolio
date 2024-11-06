from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import json
import numpy as np

app = Flask(__name__)

# Membaca laporan yang telah dihasilkan sebelumnya
daily_report = pd.read_csv('daily_report.csv', index_col=0)
weekly_report = pd.read_csv('weekly_report.csv', index_col=0)
monthly_report = pd.read_csv('monthly_report.csv', index_col=0)
quarterly_report = pd.read_csv('quarterly_report.csv', index_col=0)
semester_report = pd.read_csv('semester_report.csv', index_col=0)
yearly_report = pd.read_csv('yearly_report.csv', index_col=0)

# Membaca laporan yang telah dihasilkan sebelumnya
try:
    daily_report = pd.read_csv('daily_report.csv', index_col=0)
    print("Daily report loaded successfully")
    print(daily_report.head())
except Exception as e:
    print(f"Error loading daily report: {e}")

try:
    weekly_report = pd.read_csv('weekly_report.csv', index_col=0)
    print("Weekly report loaded successfully")
    print(weekly_report.head())
except Exception as e:
    print(f"Error loading weekly report: {e}")

try:
    monthly_report = pd.read_csv('monthly_report.csv', index_col=0)
    print("Monthly report loaded successfully")
    print(monthly_report.head())
except Exception as e:
    print(f"Error loading monthly report: {e}")

try:
    quarterly_report = pd.read_csv('quarterly_report.csv', index_col=0)
    print("Quarterly report loaded successfully")
    print(quarterly_report.head())
except Exception as e:
    print(f"Error loading quarterly report: {e}")

try:
    semester_report = pd.read_csv('semester_report.csv', index_col=0)
    print("Semester report loaded successfully")
    print(semester_report.head())
except Exception as e:
    print(f"Error loading semester report: {e}")

try:
    yearly_report = pd.read_csv('yearly_report.csv', index_col=0)
    print("Yearly report loaded successfully")
    print(yearly_report.head())
except Exception as e:
    print(f"Error loading yearly report: {e}")

# Load SVM model dan scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/report', methods=['GET', 'POST'])
def report():
    if request.method == 'POST':
        interval = request.form['interval']
        print(f"Interval selected: {interval}")  # Debugging statement
        if interval == 'Harian':
            report = daily_report
        elif interval == 'Mingguan':
            report = weekly_report
        elif interval == 'Bulanan':
            report = monthly_report
        elif interval == 'Triwulanan':
            report = quarterly_report
        elif interval == 'Semester':
            report = semester_report
        elif interval == 'Tahunan':
            report = yearly_report
        else:
            report = daily_report  # Default to daily report if interval is not recognized

        app.logger.info(report.head())  # Debugging statement to print the head of the report
        
        return render_template('report.html', tables=[report.to_html(classes='data', header="true")])
    return render_template('report.html') #redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return 'No file uploaded', 400

        # Baca data yang diunggah
        data = pd.read_csv(file)
        data = data.drop_duplicates()
        data = data.replace(['-', '---'], np.nan)
        data = data.dropna()

        # Normalisasi data
        X = data[['pm_sepuluh', 'pm_duakomalima']]
        X_scaled = scaler.transform(X)

        # Prediksi menggunakan model SVM
        predictions = model.predict(X_scaled)
        data['prediction'] = predictions
        category_map = {
            0: 'BAIK',
            1: 'SANGAT TIDAK SEHAT',
            2: 'SEDANG',
            3: 'TIDAK SEHAT'
        }
        data['kategori'] = data['prediction'].map(category_map)

        return render_template('upload.html', tables=[data.to_html(classes='data', header="true")])
    return render_template('upload.html')

@app.route('/graph', methods=['GET'])
def graph():
    interval = request.args.get('interval', 'Harian')
    if interval == 'Harian':
        report = daily_report
    elif interval == 'Mingguan':
        report = weekly_report
    elif interval == 'Bulanan':
        report = monthly_report
    elif interval == 'Triwulanan':
        report = quarterly_report
    elif interval == 'Semester':
        report = semester_report
    elif interval == 'Tahunan':
        report = yearly_report
    else:
        report = daily_report  # Default to daily report if interval is not recognized

    report = report.reset_index()
    report['index'] = report.index  # Tambahkan kolom index jika belum ada
    data = report.to_dict(orient='list')

    # Debugging statement
    print(data)  # Tambahkan ini untuk melihat data yang dikirim ke template
    
    return render_template('graph.html', data=json.dumps(data))

if __name__ == '__main__':
    app.run(debug=True)