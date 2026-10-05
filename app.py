
import streamlit as st
import joblib
from scipy.sparse import hstack, csr_matrix

# ==========================================
# Load trained models
# ==========================================

department_vectorizer = joblib.load(
    "tn_department_vectorizer.pkl"
)

department_model = joblib.load(
    "tn_department_model.pkl"
)

priority_vectorizer = joblib.load(
    "tn_priority_vectorizer.pkl"
)

priority_model = joblib.load(
    "tn_priority_model.pkl"
)

# ==========================================
# Mappings
# ==========================================

severity_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

priority_to_resolution = {
    "High": "Fast (1–7 days)",
    "Medium": "Moderate (8–20 days)",
    "Low": "Long (21–45 days)"
}

# ==========================================
# Page configuration
# ==========================================

st.set_page_config(
    page_title="Smart Civic Complaint Analyzer",
    page_icon="🏙️",
    layout="centered"
)

# ==========================================
# Title
# ==========================================

st.title("🏙️ Smart Civic Complaint Analyzer")

st.write(
    "Enter the complaint details to predict the responsible "
    "department, priority level, and expected resolution time."
)

st.divider()

# ==========================================
# Input
# ==========================================

complaint = st.text_area(
    "Complaint Description",
    placeholder="Example: Street lights are not working near the bus stand...",
    height=120
)

severity = st.selectbox(
    "Severity",
    ["Low", "Medium", "High"]
)

population = st.number_input(
    "Population Affected",
    min_value=0,
    value=100,
    step=1
)

previous_complaints = st.number_input(
    "Previous Complaints",
    min_value=0,
    value=0,
    step=1
)

# ==========================================
# Analyze
# ==========================================

if st.button("🔍 Analyze Complaint"):

    if not complaint.strip():
        st.warning("Please enter a complaint description.")

    else:

        # --------------------------------------
        # Department prediction
        # --------------------------------------

        department_vector = department_vectorizer.transform(
            [complaint]
        )

        department = department_model.predict(
            department_vector
        )[0]

        # --------------------------------------
        # Priority prediction
        # --------------------------------------

        text_vector = priority_vectorizer.transform(
            [complaint]
        )

        structured_features = csr_matrix([[
            severity_mapping[severity],
            population,
            previous_complaints
        ]])

        combined_features = hstack([
            text_vector,
            structured_features
        ])

        priority = priority_model.predict(
            combined_features
        )[0]

        # --------------------------------------
        # Resolution
        # --------------------------------------

        resolution = priority_to_resolution[priority]

        # --------------------------------------
        # Display results
        # --------------------------------------

        st.success("Complaint analyzed successfully!")

        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Department",
                department
            )

        with col2:
            st.metric(
                "Priority",
                priority
            )

        with col3:
            st.metric(
                "Expected Resolution",
                resolution
            )

        st.divider()

        st.write("### Complaint")
        st.write(complaint)
