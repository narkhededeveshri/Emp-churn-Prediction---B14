import pickle
from flask import Flask, render_template_string, request
import numpy as np

app = Flask(__name__)

# Load the trained Naive Bayes model
with open('naive_bayes.pkl', 'rb') as f:
  model = pickle.load(f)

# HTML template with Tailwind CSS styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Prediction Portal</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-slate-50 to-indigo-100 min-h-screen flex items-center justify-center p-4">
    <div class="max-w-2xl w-full bg-white shadow-2xl rounded-3xl p-8 md:p-10 border border-gray-100">
        <div class="text-center mb-8">
            <span class="bg-indigo-100 text-indigo-700 text-xs font-semibold px-3 py-1 rounded-full uppercase tracking-wider">Naive Bayes Model</span>
            <h1 class="text-3xl font-bold text-gray-900 mt-3">Employee Prediction Portal</h1>
            <p class="text-gray-500 text-sm mt-1">Enter employee details below to generate real-time predictions.</p>
        </div>

        <form method="POST" class="space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Education -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Education</label>
                    <select name="Education" required class="w-full px-4 py-2.5 rounded-xl border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition bg-white text-gray-700">
                        <option value="0">Bachelors</option>
                        <option value="1">Masters</option>
                        <option value="2">PHD</option>
                    </select>
                </div>

                <!-- Joining Year -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Joining Year</label>
                    <input type="number" name="JoiningYear" placeholder="e.g., 2017" required class="w-full px-4 py-2.5 rounded-xl border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition text-gray-700">
                </div>

                <!-- City -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">City</label>
                    <select name="City" required class="w-full px-4 py-2.5 rounded-xl border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition bg-white text-gray-700">
                        <option value="0">Bangalore</option>
                        <option value="1">New Delhi</option>
                        <option value="2">Pune</option>
                    </select>
                </div>

                <!-- Payment Tier -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Payment Tier</label>
                    <select name="PaymentTier" required class="w-full px-4 py-2.5 rounded-xl border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition bg-white text-gray-700">
                        <option value="1">Tier 1</option>
                        <option value="2">Tier 2</option>
                        <option value="3">Tier 3</option>
                    </select>
                </div>

                <!-- Age -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Age</label>
                    <input type="number" name="Age" placeholder="e.g., 28" required class="w-full px-4 py-2.5 rounded-xl border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition text-gray-700">
                </div>

                <!-- Gender -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Gender</label>
                    <select name="Gender" required class="w-full px-4 py-2.5 rounded-xl border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition bg-white text-gray-700">
                        <option value="0">Female</option>
                        <option value="1">Male</option>
                    </select>
                </div>

                <!-- Ever Benched -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Ever Benched</label>
                    <select name="EverBenched" required class="w-full px-4 py-2.5 rounded-xl border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition bg-white text-gray-700">
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>

                <!-- Experience in Current Domain -->
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">Experience in Current Domain</label>
                    <input type="number" name="ExperienceInCurrentDomain" placeholder="e.g., 3" required class="w-full px-4 py-2.5 rounded-xl border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition text-gray-700">
                </div>
            </div>

            <!-- Submit Button -->
            <div>
                <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-6 rounded-xl shadow-lg hover:shadow-indigo-500/30 transition duration-200">
                    Predict Now
                </button>
            </div>
        </form>

        {% if prediction is not none %}
        <div class="mt-8 p-5 rounded-2xl bg-indigo-50 border border-indigo-100 text-center">
            <h3 class="text-sm font-medium text-indigo-800 uppercase tracking-wide">Prediction Result</h3>
            <p class="text-2xl font-bold text-indigo-900 mt-1">
                {% if prediction == 1 %}
                    <span class="text-red-600">High Risk / Positive Outcome (1)</span>
                {% else %}
                    <span class="text-emerald-600">Low Risk / Negative Outcome (0)</span>
                {% endif %}
            </p>
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
      # Extract and convert inputs matching model features[cite: 1]
      education = float(request.form['Education'])
      joining_year = float(request.form['JoiningYear'])
      city = float(request.form['City'])
      payment_tier = float(request.form['PaymentTier'])
      age = float(request.form['Age'])
      gender = float(request.form['Gender'])
      ever_benched = float(request.form['EverBenched'])
      exp_domain = float(request.form['ExperienceInCurrentDomain'])

      features = np.array([[
          education,
          joining_year,
          city,
          payment_tier,
          age,
          gender,
          ever_benched,
          exp_domain,
      ]])

      pred = model.predict(features)
      prediction = int(pred[0])
    except Exception as e:
      print(f'Error during prediction: {e}')

  return render_template_string(HTML_TEMPLATE, prediction=prediction)


if __name__ == '__main__':
  app.run(debug=True)
