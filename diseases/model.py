import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.impute import KNNImputer
import pickle
import shap

# Load dataset
print("Loading and exploring the dataset...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=column_names)

# Replace 0s with NaN for meaningful columns
for column in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    df[column] = df[column].replace(0, np.nan)

# Use KNN Imputer for missing values
imputer = KNNImputer(n_neighbors=5)
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = imputer.fit_transform(
    df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']]
)

# Feature engineering
X = df.drop('Outcome', axis=1)
X['BMI_Age'] = X['BMI'] * X['Age']
y = df['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)

# Hyperparameter tuning using RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist, n_iter=20, cv=5, scoring='f1')
random_search.fit(X_train_scaled, y_train)
best_rf = random_search.best_estimator_

# Save model, scaler, and feature names
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(best_rf, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("Model training complete!")
