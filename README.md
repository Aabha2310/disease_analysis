# disease_analysis
# Medical Report Scanner and Disease Risk Analyzer
# Using Pima Indians Diabetes Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Dataset URL (online CSV)
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

# Load dataset
data = pd.read_csv(url)

# Features and Target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Model trained successfully\n")

# --- User Input Section ---
print("Enter Medical Report Values")

preg = int(input("Number of Pregnancies: "))
glucose = int(input("Glucose Level: "))
bp = int(input("Blood Pressure: "))
skin = int(input("Skin Thickness: "))
insulin = int(input("Insulin Level: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = int(input("Age: "))

# Create input data
input_data = [[preg, glucose, bp, skin, insulin, bmi, dpf, age]]

# Prediction
prediction = model.predict(input_data)

print("\n--- Medical Analysis Result ---")

if prediction[0] == 1:
    print("⚠ High Risk of Diabetes")
else:
    print("✓ Low Risk of Diabetes")

print("\nAnalysis Complete")
