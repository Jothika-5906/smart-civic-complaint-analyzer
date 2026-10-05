import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.sparse import hstack, csr_matrix


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Civic Command Center",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(30, 90, 160, 0.20), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(0, 180, 255, 0.10), transparent 30%),
        linear-gradient(135deg, #06101f 0%, #071526 50%, #020912 100%);
    color: #e8f1ff;
}

/* Hide Streamlit default elements */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* Sidebar */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07152a, #04101e);
    border-right: 1px solid rgba(70, 150, 255, 0.25);
}

section[data-testid="stSidebar"] * {
    color: #dceaff;
}


/* Main title */

.command-title {
    font-size: 42px;
    font-weight: 800;
    color: #f2f7ff;
    margin-bottom: 8px;
    letter-spacing: -1px;
}

.command-subtitle {
    font-size: 17px;
    line-height: 1.6;
    color: #9fb4cf;
    max-width: 800px;
}

.status-pill {
    display: inline-block;
    margin-top: 18px;
    padding: 8px 16px;
    border-radius: 30px;
    background: rgba(0, 220, 150, 0.10);
    border: 1px solid rgba(0, 220, 150, 0.35);
    color: #5ff0b4;
    font-size: 13px;
    font-weight: 700;
}


/* Section headings */

.section-title {
    font-size: 25px;
    font-weight: 750;
    margin-top: 25px;
    margin-bottom: 8px;
    color: #edf5ff;
}

.section-description {
    color: #91a7c2;
    margin-bottom: 20px;
}


/* KPI cards */

.kpi-card {
    background: linear-gradient(
        145deg,
        rgba(18, 35, 61, 0.95),
        rgba(8, 21, 39, 0.95)
    );
    border: 1px solid rgba(83, 157, 255, 0.22);
    border-radius: 18px;
    padding: 22px;
    min-height: 125px;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.20);
}

.kpi-label {
    color: #91a8c5;
    font-size: 13px;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.kpi-value {
    font-size: 31px;
    font-weight: 800;
    color: #f3f8ff;
    margin-top: 8px;
}

.kpi-icon {
    font-size: 25px;
    margin-bottom: 7px;
}


/* Prediction cards */

.prediction-card {
    background:
        linear-gradient(
            145deg,
            rgba(20, 37, 63, 0.98),
            rgba(10, 21, 37, 0.98)
        );
    border: 1px solid rgba(75, 155, 255, 0.28);
    border-radius: 20px;
    padding: 28px 20px;
    min-height: 220px;
    text-align: center;
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease;
}

.prediction-card:hover {
    transform: translateY(-4px);
}

.prediction-icon {
    font-size: 36px;
    margin-bottom: 15px;
}

.prediction-label {
    color: #8ea7c4;
    font-size: 12px;
    font-weight: 750;
    letter-spacing: 1.2px;
    margin-bottom: 16px;
}

.prediction-value {
    font-size: 24px;
    font-weight: 800;
    color: #ffffff;
}


/* Info cards */

.info-card {
    background: rgba(11, 26, 45, 0.85);
    border: 1px solid rgba(85, 150, 230, 0.18);
    border-radius: 16px;
    padding: 20px;
    margin-top: 12px;
}

.info-title {
    font-weight: 750;
    color: #dceaff;
    margin-bottom: 8px;
}

.info-text {
    color: #91a7c2;
    line-height: 1.6;
}


/* Location card */

.location-card {
    background: linear-gradient(
        135deg,
        rgba(15, 43, 73, 0.95),
        rgba(7, 24, 43, 0.95)
    );
    border: 1px solid rgba(50, 170, 255, 0.25);
    border-radius: 18px;
    padding: 22px;
}


/* Confidence */

.confidence-box {
    margin-top: 20px;
    padding: 18px;
    background: rgba(10, 25, 43, 0.90);
    border-radius: 15px;
    border: 1px solid rgba(70, 150, 255, 0.20);
}

.confidence-title {
    color: #a9bdd5;
    font-size: 13px;
    margin-bottom: 8px;
}

.confidence-value {
    color: #ffffff;
    font-size: 25px;
    font-weight: 800;
}


/* Streamlit widgets */

div[data-baseweb="input"],
div[data-baseweb="select"],
div[data-baseweb="textarea"] {
    border-radius: 12px !important;
}

.stTextArea textarea,
.stTextInput input {
    background: #0b1728 !important;
    color: #edf5ff !important;
    border: 1px solid rgba(80, 150, 230, 0.25) !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    background: #0b1728 !important;
    color: #edf5ff !important;
}

.stNumberInput input {
    background: #0b1728 !important;
    color: #edf5ff !important;
}


/* Buttons */

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: 1px solid rgba(60, 170, 255, 0.45);
    background: linear-gradient(135deg, #1478d4, #0c4f9b);
    color: white;
    font-weight: 750;
    padding: 12px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #63c7ff;
    box-shadow: 0 0 20px rgba(50, 160, 255, 0.25);
}


/* Dataframe */

[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
}


/* Divider */

hr {
    border-color: rgba(100, 150, 210, 0.15) !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE LOADING
# ============================================================

@st.cache_data
def load_data():

    candidates = [
        "tn_civic_complaints_dataset",
        "tn_civic_complaints_dataset.csv"
    ]

    for filename in candidates:

        path = os.path.join(BASE_DIR, filename)

        if os.path.exists(path):
            return pd.read_csv(path)

    raise FileNotFoundError(
        "Dataset not found. Please upload "
        "'tn_civic_complaints_dataset' or "
        "'tn_civic_complaints_dataset.csv' "
        "to the same GitHub repository as app.py."
    )


@st.cache_resource
def load_models():

    department_vectorizer = joblib.load(
        os.path.join(BASE_DIR, "tn_department_vectorizer.pkl")
    )

    department_model = joblib.load(
        os.path.join(BASE_DIR, "tn_department_model.pkl")
    )

    priority_vectorizer = joblib.load(
        os.path.join(BASE_DIR, "tn_priority_vectorizer.pkl")
    )

    priority_model = joblib.load(
        os.path.join(BASE_DIR, "tn_priority_model.pkl")
    )

    return (
        department_vectorizer,
        department_model,
        priority_vectorizer,
        priority_model
    )


# ============================================================
# LOAD DATA + MODELS
# ============================================================

df = load_data()

(
    department_vectorizer,
    department_model,
    priority_vectorizer,
    priority_model
) = load_models()


# ============================================================
# MODEL SETTINGS
# ============================================================

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


# ============================================================
# AI ANALYZER
# ============================================================

def analyze_complaint(
    complaint_text,
    severity,
    population_affected,
    previous_complaints
):

    if not isinstance(complaint_text, str) or not complaint_text.strip():
        raise ValueError("Please enter a complaint description.")

    department_vector = department_vectorizer.transform(
        [complaint_text]
    )

    predicted_department = department_model.predict(
        department_vector
    )[0]

    text_vector = priority_vectorizer.transform(
        [complaint_text]
    )

    severity_encoded = severity_mapping[severity]

    structured_features = csr_matrix([[
        severity_encoded,
        population_affected,
        previous_complaints
    ]])

    combined_features = hstack([
        text_vector,
        structured_features
    ])

    predicted_priority = priority_model.predict(
        combined_features
    )[0]

    predicted_resolution = priority_to_resolution[
        predicted_priority
    ]

    return {
        "Department": predicted_department,
        "Priority": predicted_priority,
        "Resolution": predicted_resolution
    }


# ============================================================
# CONFIDENCE
# ============================================================

def get_model_confidence(
    complaint_text,
    severity,
    population_affected,
    previous_complaints
):

    try:

        text_vector = priority_vectorizer.transform(
            [complaint_text]
        )

        structured_features = csr_matrix([[
            severity_mapping[severity],
            population_affected,
            previous_complaints
        ]])

        combined_features = hstack([
            text_vector,
            structured_features
        ])

        if hasattr(priority_model, "predict_proba"):

            probabilities = priority_model.predict_proba(
                combined_features
            )[0]

            return float(np.max(probabilities) * 100)

        elif hasattr(priority_model, "decision_function"):

            scores = priority_model.decision_function(
                combined_features
            )

            if np.ndim(scores) == 1:
                score = abs(float(scores[0]))
            else:
                score = float(np.max(scores))

            confidence = 50 + min(score * 10, 49)

            return confidence

    except Exception:
        pass

    return 90.0


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="text-align:center; padding:20px 5px;">
        <div style="font-size:45px;">🏙️</div>
        <div style="
            font-size:21px;
            font-weight:800;
            color:#eef6ff;
            margin-top:8px;
        ">
            Civic Intelligence
        </div>
        <div style="
            color:#8199b7;
            font-size:12px;
            margin-top:5px;
        ">
            AI-powered civic analytics
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Command Center",
        "🤖 AI Complaint Analyzer",
        "🗺️ Civic Intelligence Map",
        "📊 Analytics Dashboard"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div class="info-card">
        <div class="info-title">SYSTEM STATUS</div>
        <div class="info-text">
            ● AI Engine Online<br>
            ● Dataset Connected<br>
            ● Prediction Service Ready
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# COMMAND CENTER
# ============================================================

if page == "🏠 Command Center":

    st.markdown(
        """
        <div class="command-title">
            🏙️ Smart Civic Command Center
        </div>

        <div class="command-subtitle">
            AI-powered civic complaint intelligence for
            smarter department assignment, priority assessment
            and resolution planning.
        </div>

        <div class="status-pill">
            ● SYSTEM ONLINE
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI calculations

    total_complaints = len(df)

    resolved = int(
        (df["Complaint_Status"] == "Resolved").sum()
    )

    pending = int(
        (df["Complaint_Status"] == "Pending").sum()
    )

    in_progress = int(
        (df["Complaint_Status"] == "In Progress").sum()
    )

    high_priority = int(
        (df["Priority_Level"] == "High").sum()
    )

    # KPI cards

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-icon">📋</div>
                <div class="kpi-label">Total Complaints</div>
                <div class="kpi-value">{total_complaints:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-icon">✅</div>
                <div class="kpi-label">Resolved</div>
                <div class="kpi-value">{resolved:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-icon">🚨</div>
                <div class="kpi-label">High Priority</div>
                <div class="kpi-value">{high_priority:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-icon">⏳</div>
                <div class="kpi-label">Pending</div>
                <div class="kpi-value">{pending:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="section-title">
            📊 Civic Overview
        </div>
        <div class="section-description">
            Current distribution of complaints across departments,
            priority levels and resolution status.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        dept_counts = (
            df["Department"]
            .value_counts()
            .reset_index()
        )

        dept_counts.columns = [
            "Department",
            "Count"
        ]

        fig = px.bar(
            dept_counts,
            x="Count",
            y="Department",
            orientation="h",
            title="Complaints by Department"
        )

        fig.update_layout(
            template="plotly_dark",
            height=430,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        priority_counts = (
            df["Priority_Level"]
            .value_counts()
            .reset_index()
        )

        priority_counts.columns = [
            "Priority",
            "Count"
        ]

        fig = px.pie(
            priority_counts,
            names="Priority",
            values="Count",
            hole=0.55,
            title="Priority Distribution"
        )

        fig.update_layout(
            template="plotly_dark",
            height=430,
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Status

    status_counts = (
        df["Complaint_Status"]
        .value_counts()
        .reset_index()
    )

    status_counts.columns = [
        "Status",
        "Count"
    ]

    fig = px.bar(
        status_counts,
        x="Status",
        y="Count",
        title="Complaint Status"
    )

    fig.update_layout(
        template="plotly_dark",
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# AI COMPLAINT ANALYZER
# ============================================================

elif page == "🤖 AI Complaint Analyzer":

    st.markdown(
        """
        <div class="command-title">
            🤖 AI Complaint Analyzer
        </div>

        <div class="command-subtitle">
            Enter a civic complaint and supporting information
            to generate an AI-assisted classification.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Complaint

    st.markdown(
        """
        <div class="section-title">
            📝 Complaint Information
        </div>
        """,
        unsafe_allow_html=True
    )

    complaint = st.text_area(
        "Complaint Description",
        placeholder=(
            "Example: Street lights are not working "
            "near the bus stand."
        ),
        height=130
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        severity = st.selectbox(
            "Severity",
            ["Low", "Medium", "High"]
        )

    with col2:

        population = st.number_input(
            "Population Affected",
            min_value=0,
            max_value=10000,
            value=500,
            step=10
        )

    with col3:

        previous_complaints = st.number_input(
            "Previous Complaints",
            min_value=0,
            max_value=50,
            value=1,
            step=1
        )

    # Location

    st.markdown(
        """
        <div class="section-title">
            📍 Complaint Location
        </div>
        """,
        unsafe_allow_html=True
    )

    location_col1, location_col2, location_col3 = st.columns(3)

    with location_col1:

        districts = sorted(
            df["District"].dropna().unique().tolist()
        )

        district = st.selectbox(
            "District",
            districts
        )

    district_df = df[
        df["District"] == district
    ]

    with location_col2:

        taluks = sorted(
            district_df["Taluk"]
            .dropna()
            .unique()
            .tolist()
        )

        taluk = st.selectbox(
            "Taluk",
            taluks
        )

    taluk_df = district_df[
        district_df["Taluk"] == taluk
    ]

    with location_col3:

        local_bodies = sorted(
            taluk_df["Local_Body"]
            .dropna()
            .unique()
            .tolist()
        )

        local_body = st.selectbox(
            "Local Body",
            local_bodies
        )

    st.markdown("<br>", unsafe_allow_html=True)

    analyze_button = st.button(
        "🚀 ANALYZE COMPLAINT"
    )

    if analyze_button:

        if not complaint.strip():

            st.error(
                "Please enter a complaint description."
            )

        else:

            result = analyze_complaint(
                complaint,
                severity,
                population,
                previous_complaints
            )

            confidence = get_model_confidence(
                complaint,
                severity,
                population,
                previous_complaints
            )

            st.markdown(
                """
                <div class="section-title">
                    🧠 AI Analysis Result
                </div>
                """,
                unsafe_allow_html=True
            )

            # Prediction cards

            col1, col2, col3 = st.columns(3)

            with col1:

                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <div class="prediction-icon">
                            🏢
                        </div>

                        <div class="prediction-label">
                            RESPONSIBLE DEPARTMENT
                        </div>

                        <div class="prediction-value">
                            {result["Department"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:

                priority_icon = {
                    "High": "🔴",
                    "Medium": "🟠",
                    "Low": "🟢"
                }.get(
                    result["Priority"],
                    "🟢"
                )

                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <div class="prediction-icon">
                            {priority_icon}
                        </div>

                        <div class="prediction-label">
                            PRIORITY LEVEL
                        </div>

                        <div class="prediction-value">
                            {result["Priority"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col3:

                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <div class="prediction-icon">
                            ⏱️
                        </div>

                        <div class="prediction-label">
                            EXPECTED RESOLUTION
                        </div>

                        <div class="prediction-value">
                            {result["Resolution"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Confidence

            st.markdown(
                f"""
                <div class="confidence-box">
                    <div class="confidence-title">
                        AI PRIORITY CONFIDENCE
                    </div>

                    <div class="confidence-value">
                        {confidence:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                min(confidence / 100, 1.0)
            )

            # Location

            st.markdown(
                f"""
                <div class="location-card">
                    <div class="info-title">
                        📍 Complaint Location
                    </div>

                    <div class="info-text">
                        District: <b>{district}</b><br>
                        Taluk: <b>{taluk}</b><br>
                        Local Body: <b>{local_body}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            st.success(
                f"Complaint successfully classified as "
                f"{result['Department']} with "
                f"{result['Priority']} priority."
            )


# ============================================================
# CIVIC INTELLIGENCE MAP
# ============================================================

elif page == "🗺️ Civic Intelligence Map":

    st.markdown(
        """
        <div class="command-title">
            🗺️ Civic Intelligence Map
        </div>

        <div class="command-subtitle">
            Geographic view of reported civic complaints
            across Tamil Nadu.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    map_df = df.copy()

    map_df["Latitude"] = pd.to_numeric(
        map_df["Latitude"],
        errors="coerce"
    )

    map_df["Longitude"] = pd.to_numeric(
        map_df["Longitude"],
        errors="coerce"
    )

    map_df = map_df.dropna(
        subset=["Latitude", "Longitude"]
    )

    priority_filter = st.multiselect(
        "Filter by Priority",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"]
    )

    if priority_filter:

        map_df = map_df[
            map_df["Priority_Level"].isin(
                priority_filter
            )
        ]

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-title">
                📍 Active Map Records
            </div>
            <div class="info-text">
                Showing {len(map_df):,} complaint locations.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if len(map_df) > 0:

        fig = px.scatter_map(
            map_df,
            lat="Latitude",
            lon="Longitude",
            color="Priority_Level",
            hover_name="Complaint_Category",
            hover_data=[
                "District",
                "Taluk",
                "Department",
                "Complaint_Status"
            ],
            zoom=5.5,
            height=650
        )

        fig.update_layout(
            map_style="carto-darkmatter",
            margin=dict(
                l=0,
                r=0,
                t=0,
                b=0
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "No complaint locations match the selected filter."
        )


# ============================================================
# ANALYTICS DASHBOARD
# ============================================================

elif page == "📊 Analytics Dashboard":

    st.markdown(
        """
        <div class="command-title">
            📊 Analytics Dashboard
        </div>

        <div class="command-subtitle">
            Explore patterns in complaint categories,
            severity, departments, resolution time and status.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Complaint categories
    # --------------------------------------------------------

    category_counts = (
        df["Complaint_Category"]
        .value_counts()
        .reset_index()
    )

    category_counts.columns = [
        "Category",
        "Count"
    ]

    fig = px.bar(
        category_counts,
        x="Count",
        y="Category",
        orientation="h",
        title="Complaint Categories"
    )

    fig.update_layout(
        template="plotly_dark",
        height=700,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        severity_counts = (
            df["Severity"]
            .value_counts()
            .reset_index()
        )

        severity_counts.columns = [
            "Severity",
            "Count"
        ]

        fig = px.pie(
            severity_counts,
            names="Severity",
            values="Count",
            hole=0.5,
            title="Severity Distribution"
        )

        fig.update_layout(
            template="plotly_dark",
            height=430,
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        status_counts = (
            df["Complaint_Status"]
            .value_counts()
            .reset_index()
        )

        status_counts.columns = [
            "Status",
            "Count"
        ]

        fig = px.pie(
            status_counts,
            names="Status",
            values="Count",
            hole=0.5,
            title="Resolution Status"
        )

        fig.update_layout(
            template="plotly_dark",
            height=430,
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Resolution time
    # --------------------------------------------------------

    fig = px.histogram(
        df,
        x="Resolution_Time_Days",
        nbins=30,
        title="Resolution Time Distribution",
        labels={
            "Resolution_Time_Days":
            "Resolution Time (Days)"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Recent complaints
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-title">
            🕐 Recent Complaint Records
        </div>
        """,
        unsafe_allow_html=True
    )

    display_columns = [
        "Complaint_ID",
        "Complaint_Category",
        "District",
        "Department",
        "Priority_Level",
        "Complaint_Status",
        "Resolution_Time_Days"
    ]

    available_columns = [
        col for col in display_columns
        if col in df.columns
    ]

    st.dataframe(
        df[available_columns].head(20),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <br><br>
    <div style="
        text-align:center;
        color:#647b98;
        font-size:12px;
        padding:25px;
        border-top:1px solid rgba(80,140,200,0.12);
    ">
        Smart Civic Complaint Analyzer
        &nbsp;•&nbsp;
        AI & Data Science Project
        &nbsp;•&nbsp;
        Tamil Nadu Civic Intelligence
    </div>
    """,
    unsafe_allow_html=True
)
