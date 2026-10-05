 import streamlit as st
import joblib
from scipy.sparse import hstack, csr_matrix

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Civic Complaint Analyzer",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

department_vectorizer = joblib.load("tn_department_vectorizer.pkl")
department_model = joblib.load("tn_department_model.pkl")

priority_vectorizer = joblib.load("tn_priority_vectorizer.pkl")
priority_model = joblib.load("tn_priority_model.pkl")

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

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Header */

.hero {
    background: linear-gradient(135deg, #172554, #1e40af);
    padding: 32px;
    border-radius: 18px;
    color: white;
    margin-bottom: 28px;
}

.hero h1 {
    font-size: 38px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 17px;
    opacity: 0.9;
}

/* Section headings */

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

/* Cards */

.result-card {
    background: white;
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    min-height: 145px;
}

.result-icon {
    font-size: 28px;
}

.result-label {
    color: #6b7280;
    font-size: 14px;
    margin-top: 8px;
}

.result-value {
    font-size: 25px;
    font-weight: 700;
    margin-top: 6px;
}

/* Input box */

.input-card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 15px rgba(0,0,0,0.04);
}

/* Info cards */

.info-card {
    background: #eef4ff;
    padding: 20px;
    border-radius: 14px;
    border-left: 5px solid #2563eb;
}

/* Button */

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 48px;
    font-size: 16px;
    font-weight: 600;
}

/* Footer */

.footer {
    text-align: center;
    color: #6b7280;
    font-size: 13px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">

<h1>🏙️ Smart Civic Complaint Analyzer</h1>

<p>
AI-powered civic complaint classification for faster
department assignment, priority assessment and resolution planning.
</p>

</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# COMPLAINT INPUT
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📝 Complaint Details</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="input-card">', unsafe_allow_html=True)

complaint = st.text_area(
    "Complaint Description",
    placeholder=(
        "Example: Street lights are not working near the bus stand, "
        "creating a safety concern at night."
    ),
    height=130,
    label_visibility="visible"
)

col1, col2, col3 = st.columns(3)

with col1:
    severity = st.selectbox(
        "⚠️ Severity",
        ["Low", "Medium", "High"]
    )

with col2:
    population = st.number_input(
        "👥 Population Affected",
        min_value=0,
        value=100,
        step=1
    )

with col3:
    previous_complaints = st.number_input(
        "📋 Previous Complaints",
        min_value=0,
        value=0,
        step=1
    )

st.markdown("<br>", unsafe_allow_html=True)

analyze = st.button(
    "🔍 Analyze Complaint",
    type="primary"
)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

if analyze:

    if not complaint.strip():

        st.warning("Please enter a complaint description.")

    else:

        # Department prediction

        department_vector = department_vectorizer.transform(
            [complaint]
        )

        department = department_model.predict(
            department_vector
        )[0]

        # Priority prediction

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

        resolution = priority_to_resolution[priority]

        # --------------------------------------------------
        # RESULTS
        # --------------------------------------------------

        st.markdown(
            '<div class="section-title">📊 Analysis Results</div>',
            unsafe_allow_html=True
        )

        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-icon">🏢</div>
                <div class="result-label">RESPONSIBLE DEPARTMENT</div>
                <div class="result-value">{department}</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:

            priority_icon = {
                "High": "🔴",
                "Medium": "🟠",
                "Low": "🟢"
            }[priority]

            st.markdown(f"""
            <div class="result-card">
                <div class="result-icon">{priority_icon}</div>
                <div class="result-label">PRIORITY LEVEL</div>
                <div class="result-value">{priority}</div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-icon">⏱️</div>
                <div class="result-label">EXPECTED RESOLUTION</div>
                <div class="result-value">{resolution}</div>
            </div>
            """, unsafe_allow_html=True)

        st.success("Complaint analyzed successfully.")

        # --------------------------------------------------
        # COMPLAINT SUMMARY
        # --------------------------------------------------

        st.markdown(
            '<div class="section-title">📌 Complaint Summary</div>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="info-card">

        <b>Complaint:</b><br>
        {complaint}<br><br>

        <b>Severity:</b> {severity}
        &nbsp;&nbsp; | &nbsp;&nbsp;

        <b>Population Affected:</b> {population}
        &nbsp;&nbsp; | &nbsp;&nbsp;

        <b>Previous Complaints:</b> {previous_complaints}

        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# HOW IT WORKS
# --------------------------------------------------

st.markdown(
    '<div class="section-title">⚙️ How It Works</div>',
    unsafe_allow_html=True
)

h1, h2, h3, h4 = st.columns(4)

with h1:
    st.markdown("""
    **1️⃣ Complaint**

    Citizen enters the complaint description.
    """)

with h2:
    st.markdown("""
    **2️⃣ Text Processing**

    TF-IDF converts the complaint into numerical features.
    """)

with h3:
    st.markdown("""
    **3️⃣ ML Prediction**

    Machine learning models analyze the complaint.
    """)

with h4:
    st.markdown("""
    **4️⃣ Result**

    Department, priority and resolution time are displayed.
    """)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">

Smart Civic Complaint Analyzer • Principles of Data Science Project

</div>
""", unsafe_allow_html=True)
