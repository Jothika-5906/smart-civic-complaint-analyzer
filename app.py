import streamlit as st
import pandas as pd
import joblib
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
# LOAD MODELS AND DATA
# ============================================================

@st.cache_resource
def load_models():

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

    return (
        department_vectorizer,
        department_model,
        priority_vectorizer,
        priority_model
    )


@st.cache_data
def load_data():

    return pd.read_csv(
        "tn_civic_complaints_dataset.csv"
    )


(
    department_vectorizer,
    department_model,
    priority_vectorizer,
    priority_model
) = load_models()

df = load_data()


# ============================================================
# MAPPINGS
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

priority_color = {
    "High": "#ef4444",
    "Medium": "#f59e0b",
    "Low": "#22c55e"
}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(37,99,235,0.12),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(14,165,233,0.10),
            transparent 30%
        ),
        #07111f;
    color: #e5eefb;
}

/* Main container */

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}


/* Sidebar */

[data-testid="stSidebar"] {
    background: #050b15;
    border-right: 1px solid rgba(255,255,255,0.08);
}


/* Header */

.command-header {
    background:
        linear-gradient(
            135deg,
            rgba(15,23,42,0.98),
            rgba(15,48,92,0.95)
        );

    border: 1px solid rgba(96,165,250,0.25);
    border-radius: 22px;
    padding: 30px 34px;
    margin-bottom: 24px;

    box-shadow:
        0 0 35px rgba(37,99,235,0.12);
}

.command-title {
    font-size: 38px;
    font-weight: 800;
    letter-spacing: -1px;
}

.command-subtitle {
    color: #9fb4ce;
    font-size: 16px;
    margin-top: 6px;
}

.status-pill {
    display: inline-block;
    background: rgba(34,197,94,0.12);
    color: #4ade80;
    border: 1px solid rgba(34,197,94,0.30);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 13px;
    margin-top: 15px;
}


/* KPI cards */

.kpi {
    background: rgba(15,23,42,0.78);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 16px;
    padding: 20px;
    min-height: 120px;

    box-shadow:
        0 8px 25px rgba(0,0,0,0.18);
}

.kpi-icon {
    font-size: 25px;
}

.kpi-label {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 8px;
}

.kpi-value {
    font-size: 30px;
    font-weight: 800;
    margin-top: 3px;
}


/* Panels */

.panel {
    background: rgba(15,23,42,0.82);
    border: 1px solid rgba(148,163,184,0.13);
    border-radius: 18px;
    padding: 24px;
    margin-top: 22px;

    box-shadow:
        0 8px 30px rgba(0,0,0,0.16);
}


/* Section */

.section-title {
    font-size: 23px;
    font-weight: 750;
    margin-bottom: 15px;
}


/* Prediction cards */

.prediction {
    background: rgba(15,23,42,0.95);
    border-radius: 17px;
    padding: 23px;
    min-height: 150px;
    border: 1px solid rgba(96,165,250,0.16);
}

.prediction-icon {
    font-size: 30px;
}

.prediction-label {
    color: #94a3b8;
    font-size: 12px;
    margin-top: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.prediction-value {
    font-size: 25px;
    font-weight: 800;
    margin-top: 5px;
}


/* Info */

.info-box {
    background: rgba(30,64,175,0.10);
    border: 1px solid rgba(96,165,250,0.18);
    border-radius: 14px;
    padding: 17px;
}


/* Footer */

.footer {
    text-align: center;
    color: #64748b;
    padding-top: 40px;
    font-size: 13px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        font-size:24px;
        font-weight:800;
        margin-bottom:4px;">
        🏙️ Civic AI
    </div>

    <div style="
        color:#64748b;
        font-size:13px;
        margin-bottom:25px;">
        Smart City Intelligence Platform
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Command Center",
        "🤖 AI Complaint Analyzer",
        "🗺️ Civic Intelligence Map",
        "📊 Analytics Dashboard"
    ]
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    <div style="color:#64748b;font-size:12px;">
    SYSTEM STATUS
    </div>

    <div style="
        color:#4ade80;
        font-size:14px;
        font-weight:700;
        margin-top:5px;">
        ● AI ENGINE ONLINE
    </div>

    <div style="
        color:#64748b;
        font-size:12px;
        margin-top:12px;">
        Tamil Nadu Civic Dataset
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="command-header">

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

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# COMMAND CENTER
# ============================================================

if page == "🏠 Command Center":

    total_complaints = len(df)
    total_departments = df["Department"].nunique()
    total_districts = df["District"].nunique()
    resolved = (df["Complaint_Status"] == "Resolved").sum()

    st.markdown(
        '<div class="section-title">📡 Civic Intelligence Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-icon">📋</div>
                <div class="kpi-label">TOTAL COMPLAINTS</div>
                <div class="kpi-value">{total_complaints:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-icon">🏢</div>
                <div class="kpi-label">DEPARTMENTS</div>
                <div class="kpi-value">{total_departments}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-icon">📍</div>
                <div class="kpi-label">DISTRICTS</div>
                <div class="kpi-value">{total_districts}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-icon">✅</div>
                <div class="kpi-label">RESOLVED</div>
                <div class="kpi-value">{resolved:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------------------------------------
    # Charts
    # ---------------------------------------------

    left, right = st.columns(2)

    with left:

        priority_counts = df["Priority_Level"].value_counts()

        fig = px.pie(
            values=priority_counts.values,
            names=priority_counts.index,
            hole=0.62,
            title="Priority Distribution"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=390
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        department_counts = (
            df["Department"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        department_counts.columns = [
            "Department",
            "Complaints"
        ]

        fig = px.bar(
            department_counts,
            x="Complaints",
            y="Department",
            orientation="h",
            title="Top Departments by Complaint Volume"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=390
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------------------------------------
    # Recent complaints
    # ---------------------------------------------

    st.markdown(
        '<div class="section-title">🛰️ Recent Civic Activity</div>',
        unsafe_allow_html=True
    )

    recent = df[
        [
            "Complaint_ID",
            "Complaint_Category",
            "District",
            "Priority_Level",
            "Complaint_Status"
        ]
    ].head(10)

    st.dataframe(
        recent,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# AI COMPLAINT ANALYZER
# ============================================================

elif page == "🤖 AI Complaint Analyzer":

    st.markdown(
        '<div class="section-title">🤖 AI Complaint Analyzer</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter a civic complaint and supporting information "
        "to generate an AI-assisted classification."
    )

    # ---------------------------------------------
    # Complaint
    # ---------------------------------------------

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    complaint = st.text_area(
        "📝 Complaint Description",
        placeholder=(
            "Example: Street lights are not working near "
            "the bus stand, creating a safety concern at night."
        ),
        height=130
    )

    st.markdown("### 📍 Complaint Location")

    col1, col2, col3 = st.columns(3)

    with col1:

        districts = sorted(
            df["District"].dropna().unique()
        )

        district = st.selectbox(
            "District",
            districts
        )

    district_data = df[
        df["District"] == district
    ]

    with col2:

        taluks = sorted(
            district_data["Taluk"]
            .dropna()
            .unique()
        )

        taluk = st.selectbox(
            "Taluk",
            taluks
        )

    taluk_data = district_data[
        district_data["Taluk"] == taluk
    ]

    with col3:

        local_bodies = sorted(
            taluk_data["Local_Body"]
            .dropna()
            .unique()
        )

        local_body = st.selectbox(
            "Local Body",
            local_bodies
        )

    st.markdown("### 📊 Complaint Context")

    c1, c2, c3 = st.columns(3)

    with c1:

        severity = st.selectbox(
            "⚠️ Severity",
            ["Low", "Medium", "High"]
        )

    with c2:

        population = st.number_input(
            "👥 Population Affected",
            min_value=0,
            value=100,
            step=1
        )

    with c3:

        previous_complaints = st.number_input(
            "📋 Previous Complaints",
            min_value=0,
            value=0,
            step=1
        )

    analyze = st.button(
        "🚀 RUN AI ANALYSIS",
        type="primary",
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------
    # Location map preview
    # ---------------------------------------------

    location_data = taluk_data[
        taluk_data["Local_Body"] == local_body
    ]

    if len(location_data) > 0:

        map_data = location_data[
            ["Latitude", "Longitude"]
        ].dropna()

        if len(map_data) > 0:

            st.markdown(
                '<div class="section-title">📍 Selected Civic Zone</div>',
                unsafe_allow_html=True
            )

            st.map(
                map_data,
                latitude="Latitude",
                longitude="Longitude",
                zoom=9
            )

    # ---------------------------------------------
    # AI analysis
    # ---------------------------------------------

    if analyze:

        if not complaint.strip():

            st.warning(
                "Please enter a complaint description."
            )

        else:

            # Department

            department_vector = (
                department_vectorizer.transform(
                    [complaint]
                )
            )

            department = department_model.predict(
                department_vector
            )[0]

            # Department confidence

            if hasattr(
                department_model,
                "predict_proba"
            ):

                department_probability = (
                    department_model.predict_proba(
                        department_vector
                    )[0]
                )

                department_confidence = (
                    department_probability.max() * 100
                )

            else:

                department_confidence = 0


            # Priority

            text_vector = (
                priority_vectorizer.transform(
                    [complaint]
                )
            )

            structured_features = csr_matrix(
                [[
                    severity_mapping[severity],
                    population,
                    previous_complaints
                ]]
            )

            combined_features = hstack(
                [
                    text_vector,
                    structured_features
                ]
            )

            priority = priority_model.predict(
                combined_features
            )[0]

            # Priority confidence

            if hasattr(
                priority_model,
                "predict_proba"
            ):

                priority_probability = (
                    priority_model.predict_proba(
                        combined_features
                    )[0]
                )

                priority_confidence = (
                    priority_probability.max() * 100
                )

            else:

                priority_confidence = 0

            resolution = priority_to_resolution[
                priority
            ]

            # -----------------------------------------
            # Results
            # -----------------------------------------

            st.markdown(
                '<div class="section-title">🧠 AI Analysis Result</div>',
                unsafe_allow_html=True
            )

            r1, r2, r3 = st.columns(3)

            with r1:

                st.markdown(
                    f"""
                    <div class="prediction">

                        <div class="prediction-icon">
                            🏢
                        </div>

                        <div class="prediction-label">
                            RESPONSIBLE DEPARTMENT
                        </div>

                        <div class="prediction-value">
                            {department}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with r2:

                icon = {
                    "High": "🔴",
                    "Medium": "🟠",
                    "Low": "🟢"
                }[priority]

                st.markdown(
                    f"""
                    <div class="prediction">

                        <div class="prediction-icon">
                            {icon}
                        </div>

                        <div class="prediction-label">
                            PRIORITY LEVEL
                        </div>

                        <div class="prediction-value">
                            {priority}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with r3:

                st.markdown(
                    f"""
                    <div class="prediction">

                        <div class="prediction-icon">
                            ⏱️
                        </div>

                        <div class="prediction-label">
                            EXPECTED RESOLUTION
                        </div>

                        <div class="prediction-value">
                            {resolution}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # -----------------------------------------
            # Confidence
            # -----------------------------------------

            st.markdown(
                '<div class="panel">',
                unsafe_allow_html=True
            )

            st.markdown(
                "### 🎯 Model Confidence"
            )

            conf1, conf2 = st.columns(2)

            with conf1:

                st.write(
                    f"Department: "
                    f"**{department_confidence:.1f}%**"
                )

                st.progress(
                    min(
                        int(department_confidence),
                        100
                    )
                )

            with conf2:

                st.write(
                    f"Priority: "
                    f"**{priority_confidence:.1f}%**"
                )

                st.progress(
                    min(
                        int(priority_confidence),
                        100
                    )
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            # -----------------------------------------
            # Complaint summary
            # -----------------------------------------

            st.markdown(
                '<div class="panel">',
                unsafe_allow_html=True
            )

            st.markdown(
                "### 📌 Complaint Intelligence"
            )

            st.markdown(
                f"""
                <div class="info-box">

                <b>Location:</b>
                {district} → {taluk} → {local_body}

                <br><br>

                <b>Severity:</b> {severity}

                &nbsp;&nbsp; | &nbsp;&nbsp;

                <b>Population Affected:</b> {population}

                &nbsp;&nbsp; | &nbsp;&nbsp;

                <b>Previous Complaints:</b>
                {previous_complaints}

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


# ============================================================
# CIVIC INTELLIGENCE MAP
# ============================================================

elif page == "🗺️ Civic Intelligence Map":

    st.markdown(
        '<div class="section-title">🗺️ Civic Intelligence Map</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Explore complaint locations across Tamil Nadu."
    )

    map_df = df[
        [
            "Complaint_ID",
            "Complaint_Category",
            "District",
            "Priority_Level",
            "Latitude",
            "Longitude"
        ]
    ].dropna(
        subset=["Latitude", "Longitude"]
    )

    # ---------------------------------------------
    # Filters
    # ---------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        priority_filter = st.multiselect(
            "Priority",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )

    with c2:

        district_filter = st.multiselect(
            "District",
            sorted(df["District"].unique())
        )

    filtered_map = map_df[
        map_df["Priority_Level"].isin(
            priority_filter
        )
    ]

    if district_filter:

        filtered_map = filtered_map[
            filtered_map["District"].isin(
                district_filter
            )
        ]

    # ---------------------------------------------
    # Map
    # ---------------------------------------------

    fig = px.scatter_map(
        filtered_map,
        lat="Latitude",
        lon="Longitude",
        color="Priority_Level",
        hover_name="Complaint_ID",
        hover_data=[
            "Complaint_Category",
            "District"
        ],
        zoom=6,
        height=650
    )

    fig.update_layout(
        map_style="open-street-map",
        margin={
            "r": 0,
            "t": 0,
            "l": 0,
            "b": 0
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "🔴 High priority   🟠 Medium priority   "
        "🟢 Low priority"
    )


# ============================================================
# ANALYTICS DASHBOARD
# ============================================================

elif page == "📊 Analytics Dashboard":

    st.markdown(
        '<div class="section-title">📊 Civic Analytics Dashboard</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Explore patterns in the Tamil Nadu civic complaint dataset."
    )

    # ---------------------------------------------
    # Row 1
    # ---------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        severity_counts = (
            df["Severity"]
            .value_counts()
            .reset_index()
        )

        severity_counts.columns = [
            "Severity",
            "Complaints"
        ]

        fig = px.bar(
            severity_counts,
            x="Severity",
            y="Complaints",
            title="Severity Distribution",
            text="Complaints"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        status_counts = (
            df["Complaint_Status"]
            .value_counts()
            .reset_index()
        )

        status_counts.columns = [
            "Status",
            "Complaints"
        ]

        fig = px.pie(
            status_counts,
            names="Status",
            values="Complaints",
            hole=0.55,
            title="Complaint Status"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------------------------------------
    # Row 2
    # ---------------------------------------------

    category_counts = (
        df["Complaint_Category"]
        .value_counts()
        .head(15)
        .reset_index()
    )

    category_counts.columns = [
        "Category",
        "Complaints"
    ]

    fig = px.bar(
        category_counts,
        x="Complaints",
        y="Category",
        orientation="h",
        title="Top 15 Complaint Categories"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=550
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------------------------
    # Resolution analysis
    # ---------------------------------------------

    fig = px.box(
        df,
        x="Priority_Level",
        y="Resolution_Time_Days",
        color="Priority_Level",
        title="Resolution Time by Priority"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    🏙️ Smart Civic Command Center
    &nbsp;•&nbsp;
    Principles of Data Science Project
    &nbsp;•&nbsp;
    AI + NLP + Civic Intelligence

    </div>
    """,
    unsafe_allow_html=True
)
