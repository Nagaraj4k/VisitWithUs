import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# --- Configuration ---
# Update this with your actual Hugging Face Repo ID
REPO_ID = "Nagaraj4k/Wellness-Tourism-Model" 
MODEL_FILENAME = "wellness_tourism_model_v1.joblib"

@st.cache_resource
def load_model():
    # Automatically downloads the model from your HF Hub
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    return joblib.load(model_path)

model = load_model()

# --- Streamlit UI ---
st.set_page_config(page_title="Visit with Us - Wellness Package", layout="wide")

st.title("🏨 Wellness Tourism Purchase Predictor")
st.write("""
This application predicts whether a customer will purchase the newly introduced **Wellness Tourism Package**. 
Enter the customer profile details below to generate a prediction.
""")

# --- Input Form ---
with st.form("prediction_form"):
    st.subheader("Customer Profile & Demographics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Female", "Male"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        monthly_income = st.number_input("Monthly Income", min_value=0.0, value=25000.0)
        
    with col2:
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        passport = st.radio("Has Passport?", ["No", "Yes"])

    with col3:
        own_car = st.radio("Owns a Car?", ["No", "Yes"])
        num_person = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
        num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)

    st.divider()
    st.subheader("Sales Engagement Details")
    col4, col5 = st.columns(2)
    
    with col4:
        contact_type = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
        pitch_duration = st.number_input("Duration of Pitch (min)", min_value=0, value=15)
        property_star = st.slider("Preferred Property Star", 3, 5, 3)

    with col5:
        num_trips = st.number_input("Number of Trips", min_value=1, value=3)
        num_followups = st.number_input("Number of Follow-ups", min_value=0, value=3)
        satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)

    submit = st.form_submit_button("Predict Purchase")

# --- Encoding & Prediction ---
if submit:
    # Logic to map categories back to numbers if your model expects numeric encoding
    # (Matches the encoding used in your training script)
    input_data = pd.DataFrame([{
        'Age': float(age),
        'TypeofContact': 1 if contact_type == "Self Enquiry" else 0,
        'CityTier': city_tier,
        'DurationOfPitch': float(pitch_duration),
        'Occupation': ["Salaried", "Small Business", "Large Business", "Free Lancer"].index(occupation),
        'Gender': 1 if gender == "Male" else 0,
        'NumberOfPersonVisiting': num_person,
        'NumberOfFollowups': float(num_followups),
        'ProductPitched': ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"].index(product_pitched),
        'PreferredPropertyStar': float(property_star),
        'MaritalStatus': ["Single", "Married", "Divorced", "Unmarried"].index(marital_status),
        'NumberOfTrips': float(num_trips),
        'Passport': 1 if passport == "Yes" else 0,
        'PitchSatisfactionScore': satisfaction,
        'OwnCar': 1 if own_car == "Yes" else 0,
        'NumberOfChildrenVisiting': float(num_children),
        'Designation': ["Executive", "Manager", "Senior Manager", "AVP", "VP"].index(designation),
        'MonthlyIncome': float(monthly_income)
    }])

    # Make prediction
    prediction_proba = model.predict_proba(input_data)[:, 1][0]
    prediction = 1 if prediction_proba >= 0.45 else 0

    # Display results
    st.divider()
    if prediction == 1:
        st.success(f"### Result: Likely to Purchase (Score: {prediction_proba:.2f})")
        st.balloons()
        st.write("This customer is a high-potential lead for the Wellness Package.")
    else:
        st.warning(f"### Result: Unlikely to Purchase (Score: {prediction_proba:.2f})")
        st.write("Target this customer with follow-up engagement or different packages.")
