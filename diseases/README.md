# Diabetes Prediction Web Application

## Project Title

Diabetes Prediction Web Application

## Project Description

This project is a Flask web application that predicts diabetes outcomes based on user input. It utilizes a Random Forest model trained on a dataset of diabetes-related features. The application provides SHAP plots for model interpretability and visualizes feature importance, allowing users to understand the factors influencing predictions. This tool is designed for healthcare professionals and researchers interested in diabetes prediction and analysis.

## Installation Instructions

1. **Prerequisites**:

   - Python 3.x
   - Required libraries: Flask, NumPy, Pandas, Scikit-learn, SHAP, Matplotlib, Seaborn

2. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application**:

   ```bash
   python app.py
   ```

2. **Access the web interface**:
   Open your web browser and navigate to `http://127.0.0.1:5000/`. You can input data for predictions through the provided form.

3. **Example Input Data**:
   - Pregnancies: 2
   - Glucose: 120
   - Blood Pressure: 70
   - Skin Thickness: 20
   - Insulin: 80
   - BMI: 30
   - Diabetes Pedigree Function: 0.5
   - Age: 25

## Model Details

- The model is trained on a dataset sourced from the Pima Indians Diabetes Database.
- Data preprocessing includes handling missing values using KNN Imputer and feature engineering to create new features.
- The Random Forest model is selected for its performance in classification tasks.
- Performance metrics:
  - Accuracy: 85%
  - Precision: 80%
  - Recall: 75%
  - F1 Score: 77%

## Visualizations

- The application generates SHAP plots to explain the model's predictions. These plots show the contribution of each feature to the prediction for individual instances.
- Feature importance plots visualize which features are most influential in the model's decision-making process.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
