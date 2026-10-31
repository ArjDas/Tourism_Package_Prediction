
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

st.title("Wellness Tourism Package Prediction")
st.write("""
This application predicts whether a customer is likely to purchase
the Wellness Tourism Package based on their profile and travel preferences.
Enter the customer details below to get a prediction.
""")

# Customer details
age = st.number_input("Age", 18, 100, 30)
city_tier = st.selectbox("City Tier", [1, 2, 3])

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Freelancer"]
)

gender = st.selectbox("Gender", ["Male", "Female"])

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

# Travel preferences
number_of_person_visiting = st.number_input(
    "Number of Persons Visiting", 1, 10, 2
)

preferred_property_star = st.selectbox(
    "Preferred Property Star", [3, 4, 5]
)

number_of_trips = st.number_input(
    "Number of Trips Annually", 1, 30, 3
)

passport = st.selectbox("Passport", [0, 1])

own_car = st.selectbox("Own Car", [0, 1])

number_of_children_visiting = st.number_input(
    "Number of Children Visiting", 0, 10, 0
)

monthly_income = st.number_input(
    "Monthly Income", 1000, 200000, 25000
)

# Customer interaction details
type_of_contact = st.selectbox(
    "Type of Contact", ["Company Invited", "Self Inquiry"]
)

pitch_satisfaction_score = st.selectbox(
    "Pitch Satisfaction Score", [1, 2, 3, 4, 5]
)

product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
)

number_of_followups = st.number_input(
    "Number of Followups", 1, 10, 3
)

duration_of_pitch = st.number_input(
    "Duration of Pitch (Minutes)", 0.0, 150.0, 10.0
)

# Prepare input data
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": number_of_followups,
    "DurationOfPitch": duration_of_pitch
}])

if st.button("Predict Purchase"):

    prediction = model.predict(input_data)[0]

    result = (
        "Likely to Purchase Wellness Tourism Package"
        if prediction == 1
        else "Unlikely to Purchase Wellness Tourism Package"
    )

    st.subheader("Prediction Result:")

    if prediction == 1:
        st.success(f"The model predicts: **{result}**")
    else:
        st.warning(f"The model predicts: **{result}**")
