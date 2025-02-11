import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize LabelEncoders for categorical data
age_encoder = LabelEncoder()
gender_encoder = LabelEncoder()
education_level_encoder = LabelEncoder()
job_title_encoder = LabelEncoder()
years_experience_encoder = LabelEncoder()

# Example categorical mappings (these should match the model training data)
gender_encoder.fit(["Male", "Female"])
education_level_encoder.fit(["High School", "Bachelor's", "Master's", "PhD"])
job_title_encoder.fit(["Engineer", "Manager", "Analyst", "Sales", "HR"])

app = Flask(__name__)

# Home page route
@app.route("/")
def home():
    return render_template("index.html")  # Web interface

# API endpoint for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input from form
        age = request.form["age"]  # Age as text
        gender = request.form["gender"]
        education_level = request.form["education_level"]
        job_title = request.form["job_title"]
        years_experience = request.form["years_experience"]
        
        # Encode categorical values
        gender_encoded = gender_encoder.transform([gender])[0]
        education_level_encoded = education_level_encoder.transform([education_level])[0]
        job_title_encoded = job_title_encoder.transform([job_title])[0]
        
        features = np.array([age, gender_encoded, education_level_encoded, job_title_encoded, years_experience]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        return render_template("index.html", prediction_text=f"Predicted Salary: {prediction[0]} INR")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)


