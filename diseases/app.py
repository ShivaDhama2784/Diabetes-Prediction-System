from flask import Flask, render_template, url_for, request, jsonify
import numpy as np
import pickle
import shap
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Create Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Load model, scaler, and feature names
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Load SHAP explainer once globally
explainer = shap.Explainer(model)

# Reusable functions for generating plots
def generate_shap_plot(shap_values, features_scaled):
    shap_image = BytesIO()
    shap.summary_plot(shap_values, features_scaled, show=False, feature_names=feature_names)
    plt.savefig(shap_image, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    return base64.b64encode(shap_image.getvalue()).decode()

def generate_importance_plot():
    feature_importance = model.feature_importances_
    importance_image = BytesIO()
    plt.barh(feature_names, feature_importance)
    plt.savefig(importance_image, format='png', dpi=300)
    plt.close()
    return base64.b64encode(importance_image.getvalue()).decode()

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Custom static route
@app.route('/static/<path:filename>')
def custom_static(filename):
    return app.send_static_file(filename)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        features = [float(request.form.get(feature, 0)) for feature in feature_names]
        features_array = np.array(features).reshape(1, -1)

        # Scale data and predict
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)[0][1] * 100

        # Generate SHAP values and plots
        shap_values = explainer(features_scaled)
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': probability,
            'shap_plot': generate_shap_plot(shap_values, features_scaled),
            'force_plot': generate_importance_plot(),
        })

    except ValueError:
        return jsonify({'error': 'Invalid input detected!'}), 400
    except Exception as e:
        return jsonify({'error': f'Something went wrong: {str(e)}'}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
