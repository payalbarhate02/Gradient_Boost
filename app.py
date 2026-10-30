from flask import Flask, render_template_string, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained Gradient Boosting model
with open('gradient.pkl', 'rb') as f:
    model = pickle.load(f)

# Modern and attractive HTML template styled with Tailwind CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Insurance Cost Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen flex items-center justify-center p-4">
    <div class="max-w-xl w-full bg-white rounded-2xl shadow-xl overflow-hidden p-8">
        <div class="text-center mb-8">
            <h1 class="text-3xl font-extrabold text-indigo-900">Insurance Cost Predictor</h1>
            <p class="text-gray-500 text-sm mt-2">Enter your details below to estimate your medical insurance charges.</p>
        </div>

        <form method="POST" class="space-y-5">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <!-- Age -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Age</label>
                    <input type="number" name="age" required min="1" max="120" value="{{ request.form.get('age', 25) }}"
                        class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition">
                </div>

                <!-- BMI -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">BMI</label>
                    <input type="number" step="0.1" name="bmi" required min="10" max="60" value="{{ request.form.get('bmi', 22.5) }}"
                        class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition">
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <!-- Children -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Children / Dependents</label>
                    <input type="number" name="children" required min="0" max="10" value="{{ request.form.get('children', 0) }}"
                        class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition">
                </div>

                <!-- Sex -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Sex</label>
                    <select name="sex" class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition">
                        <option value="1" {% if request.form.get('sex') == '1' %}selected{% endif %}>Male</option>
                        <option value="0" {% if request.form.get('sex') == '0' %}selected{% endif %}>Female</option>
                    </select>
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <!-- Smoker -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Smoker</label>
                    <select name="smoker" class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition">
                        <option value="1" {% if request.form.get('smoker') == '1' %}selected{% endif %}>Yes</option>
                        <option value="0" {% if request.form.get('smoker') == '0' %}selected{% endif %}>No</option>
                    </select>
                </div>

                <!-- Region -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Region</label>
                    <select name="region" class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition">
                        <option value="0" {% if request.form.get('region') == '0' %}selected{% endif %}>Northeast</option>
                        <option value="1" {% if request.form.get('region') == '1' %}selected{% endif %}>Northwest</option>
                        <option value="2" {% if request.form.get('region') == '2' %}selected{% endif %}>Southeast</option>
                        <option value="3" {% if request.form.get('region') == '3' %}selected{% endif %}>Southwest</option>
                    </select>
                </div>
            </div>

            <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 rounded-lg shadow-md transition duration-200">
                Predict Charges
            </button>
        </form>

        {% if prediction is not none %}
        <div class="mt-8 p-4 bg-indigo-50 border border-indigo-200 rounded-xl text-center animate-pulse">
            <h2 class="text-sm font-semibold text-indigo-600 uppercase tracking-wider">Estimated Insurance Cost</h2>
            <p class="text-3xl font-bold text-indigo-900 mt-1">${{ "%.2f"|format(prediction) }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract inputs in the order required by the model: age, sex, bmi, children, smoker, region
            age = float(request.form['age'])
            sex = float(request.form['sex'])
            bmi = float(request.form['bmi'])
            children = float(request.form['children'])
            smoker = float(request.form['smoker'])
            region = float(request.form['region'])

            features = np.array([[age, sex, bmi, children, smoker, region]])
            prediction = model.predict(features)[0]
        except Exception as e:
            print("Error during prediction:", e)
            prediction = None

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
