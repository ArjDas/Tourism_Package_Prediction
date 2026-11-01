
import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(
    os.path.dirname(__file__),
    "best_tourism_model_v1.joblib"
)
model = joblib.load(model_path)

st.title("Tourism Package Purchase Prediction App")

st.write("""
This application predicts whether a customer is likely to purchase
a tourism package based on their profile and interaction details.
Enter the customer information below to get a prediction.
""")

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

Age = st.number_input("Age", 18, 100, 30)
CityTier = st.selectbox("City Tier", [1, 2, 3])

DurationOfPitch = st.number_input(
    "Duration of Pitch (minutes)", 0, 300, 15
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business"]
)

Gender = st.selectbox("Gender", ["Male", "Female"])

NumberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting", 1, 20, 2
)

NumberOfFollowups = st.number_input(
    "Number of Followups", 0, 20, 3
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star", [3, 4, 5]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Married", "Single", "Divorced"]
)

NumberOfTrips = st.number_input("Number of Trips", 0, 50, 3)

Passport = st.selectbox("Has Passport?", [0, 1])

PitchSatisfactionScore = st.selectbox(
    "Pitch Satisfaction Score", [1, 2, 3, 4, 5]
)

OwnCar = st.selectbox("Owns a Car?", [0, 1])

NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting", 0, 10, 0
)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

MonthlyIncome = st.number_input(
    "Monthly Income", 0, 200000, 25000
)

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}])

if st.button("Predict Package Purchase"):

    prediction_proba = model.predict_proba(input_data)[0][1]
    prediction = int(prediction_proba >= 0.45)

    result = (
        "Likely to Purchase Tourism Package"
        if prediction == 1
        else "Unlikely to Purchase Tourism Package"
    )

    st.subheader("Prediction Result:")
    st.success(f"Model Prediction: **{result}**")
