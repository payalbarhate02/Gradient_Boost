import os
import pickle
from flask import Flask, render_template_string, request
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained Gradient Boosting model
model_path = "gradient.pkl"
with open(model_path, "rb") as f:
  model = pickle.load(f)

# Modern and attractive inline HTML/CSS layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gradient Boosting Model Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f1f5f9;
            --card-bg: #ffffff;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border: #cbd5e1;
        }
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }
        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            background: var(--card-bg);
            width: 100%;
            max-width: 620px;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
        }
        h2 {
            font-weight: 700;
            font-size: 26px;
            margin-bottom: 6px;
            color: var(--text-main);
            text-align: center;
        }
        .subtitle {
            text-align: center;
            color: var(--text-muted);
            font-size: 14px;
            margin-bottom: 30px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        .form-group.full-width {
            grid-column: span 2;
        }
        label {
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 6px;
            color: var(--text-main);
        }
        input, select {
            padding: 11px 14px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
            background-color: #fff;
        }
        input:focus, select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }
        button {
            width: 100%;
            padding: 13px;
            background-color: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 24px;
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        .result-card {
            margin-top: 24px;
            padding: 18px;
            background-color: #e0e7ff;
            border: 1px solid #c7d2fe;
            border-radius: 8px;
            text-align: center;
            color: #312e81;
            font-weight: 600;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Gradient Boosting Predictor</h2>
        <p class="subtitle">Provide feature inputs below to obtain model predictions</p>
        
        <form method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="Department">Department</label>
                    <input type="text" id="Department" name="Department" required placeholder="e.g., IT, HR">
                </div>
                
                <div class="form-group">
                    <label for="Experience_Years">Experience (Years)</label>
                    <input type="number" step="any" id="Experience_Years" name="Experience_Years" required placeholder="e.g., 3.5">
                </div>
                
                <div class="form-group">
                    <label for="Education_Level">Education Level</label>
                    <input type="text" id="Education_Level" name="Education_Level" required placeholder="e.g., Bachelor's">
                </div>
                
                <div class="form-group">
                    <label for="Age">Age</label>
                    <input type="number" id="Age" name="Age" required placeholder="e.g., 29">
                </div>
                
                <div class="form-group">
                    <label for="Gender">Gender</label>
                    <select id="Gender" name="Gender" required>
                        <option value="" disabled selected>Select Gender</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                        <option value="Other">Other</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="City">City</label>
                    <input type="text" id="City" name="City" required placeholder="e.g., New York">
                </div>
            </div>
            
            <button type="submit">Run Prediction</button>
        </form>

        {% if prediction is not none %}
        <div class="result-card">
            Predicted Value: {{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
  prediction = None
  if request.method == "POST":
    try:
      # Extract form inputs matching model features
      department = request.form["Department"]
      experience_years = float(request.form["Experience_Years"])
      education_level = request.form["Education_Level"]
      age = float(request.form["Age"])
      gender = request.form["Gender"]
      city = request.form["City"]

      # Construct DataFrame matching expected feature names and order
      input_data = pd.DataFrame([[
          department,
          experience_years,
          education_level,
          age,
          gender,
          city,
      ]],
                                columns=[
                                    "Department",
                                    "Experience_Years",
                                    "Education_Level",
                                    "Age",
                                    "Gender",
                                    "City",
                                ])

      # Predict using the loaded model
      pred = model.predict(input_data)
      prediction = round(float(pred[0]), 4)
    except Exception as e:
      prediction = f"Error: {str(e)}"

  return render_template_string(HTML_TEMPLATE, prediction=prediction)


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
