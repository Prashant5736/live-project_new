from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import os

app = Flask(__name__)
csv_file_path = os.path.join(os.path.dirname(__file__), 'data', 'Admission_Predict.csv')

# Load the CSV file
df = pd.read_csv(csv_file_path)
# Load dataset

# Drop unnecessary columns
df = df.drop(columns=["Serial No."], errors='ignore')

# Rename columns if necessary
df.rename(columns={'Chance of Admit ': 'Chance of Admit'}, inplace=True)

# Define features and target
X = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA']]
y = (df['Chance of Admit'] >= 0.75).astype(int)  # Convert to binary (Admitted or Not)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_scaled, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from the form
        gre = float(request.form['gre'])
        toefl = float(request.form['toefl'])
        rating = float(request.form['rating'])
        sop = float(request.form['sop'])
        lor = float(request.form['lor'])
        cgpa = float(request.form['cgpa'])

        print(f"Received input - GRE: {gre}, TOEFL: {toefl}, CGPA: {cgpa}")

        # Convert input into a NumPy array and scale it
        user_data = np.array([[gre, toefl, rating, sop, lor, cgpa]])
        user_data_scaled = scaler.transform(user_data)

        # Predict admission status
        chance_of_admit = model.predict_proba(user_data_scaled)[0][1]  # Probability of admission
        status = "Admitted" if chance_of_admit >= 0.75 else "Not Admitted"

        return render_template('result.html', chance=round(chance_of_admit, 2), status=status)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
