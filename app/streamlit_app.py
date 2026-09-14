from pathlib import Path
import sys

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.llm_recommendations import generate_llm_strategy

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "featured_customer_data.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_churn_model.joblib"
)

SHAP_PATH = (
    BASE_DIR
    / "reports"
    / "powerbi"
    / "customer_shap_explanations.csv"
)

RECOMMENDATIONS_PATH = (
    BASE_DIR
    / "reports"
    / "powerbi"
    / "customer_recommendations.csv"
)

MODEL_RESULTS_PATH = (
    BASE_DIR
    / "reports"
    / "model_results"
    / "model_comparison.csv"
)

MODEL_TEST_PATH = (
    BASE_DIR
    / "reports"
    / "powerbi"
    / "model_test_predictions.csv"
)

# ============================================================
# LOAD DATA AND MODEL
# ============================================================

@st.cache_data
def load_customer_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_shap_data():
    return pd.read_csv(SHAP_PATH)

@st.cache_data
def load_recommendations_data():
    return pd.read_csv(RECOMMENDATIONS_PATH)

@st.cache_resource
def load_churn_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_model_results():
    return pd.read_csv(MODEL_RESULTS_PATH)

@st.cache_data
def load_model_test_data():
    return pd.read_csv(MODEL_TEST_PATH)


customer_data = load_customer_data()
shap_data = load_shap_data()
recommendations_data = load_recommendations_data()
model_results = load_model_results()
model_test_data = load_model_test_data()
churn_model = load_churn_model()

# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(probability):

    if probability >= 0.70:
        return "HIGH RISK"

    elif probability >= 0.40:
        return "MEDIUM RISK"

    else:
        return "LOW RISK"

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    
    /* Customer profile cards */

    .profile-card {
        background-color: #FFFFFF;
        border: 1px solid #E4E7EC;
        border-radius: 12px;
        padding: 18px;
        min-height: 95px;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.04);
    }

    .profile-label {
        font-size: 13px;
        color: #667085;
        font-weight: 500;
        margin-bottom: 8px;
    }

    .profile-value {
        font-size: 22px;
        color: #172B4D;
        font-weight: 700;
        line-height: 1.25;
        word-wrap: break-word;
    }

    /* Main background */
    .stApp {
        background-color: #F5F7FB;
    }

    /* Main container spacing */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px;
    }

    /* Main title */
    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #172B4D;
        line-height: 1.25 !important;
        padding-top: 0.25rem;
        margin-top: 0 !important;
        margin-bottom: 4px;
    }

    /* Subtitle */
    .subtitle {
        font-size: 16px;
        color: #667085;
        margin-top: 4px;
        margin-bottom: 25px;
    }

    /* Section titles */
    .section-title {
        font-size: 20px;
        font-weight: 600;
        color: #172B4D;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    /* Generic card */
    .custom-card {
        color: #344054;
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E4E7EC;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.04);
        margin-bottom: 15px;
    }
    .custom-card b {
        color: #172B4D;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #98A2B3;
        font-size: 13px;
        margin-top: 40px;
    }
    
    /*  =========================================================
        STREAMLIT KPI METRIC CARDS
        ========================================================= */

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E4E7EC;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.04);
        overflow: visible !important;
    }

    /* Metric label */
    [data-testid="stMetricLabel"] {
        color: #667085 !important;
    }

    [data-testid="stMetricLabel"] p {
        color: #667085 !important;
        font-weight: 500;
    }

    /* Metric value */
    [data-testid="stMetricValue"] {
        color: #172B4D !important;
        overflow: visible !important;
    }

    [data-testid="stMetricValue"] div {
        color: #172B4D !important;
        font-weight: 700; !important;
        font-size: 27px !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;    
    }
    
    /* =========================================================
       FORM / SELECTBOX LABELS
       ========================================================= */

    [data-testid="stWidgetLabel"] p {
        color: #344054 !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    [data-testid="stSelectbox"] label p {
        color: #344054 !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }
    
    /*  =========================================================
        RETENTION RECOMMENDATIONS
        ========================================================= */

    .recommendation-card {
        background-color: #FFFFFF;
        border: 1px solid #E4E7EC;
        border-left: 5px solid #2563EB;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.04);
    }

    .recommendation-number {
        font-size: 12px;
        font-weight: 700;
        color: #2563EB;
        margin-bottom: 5px;
    }

    .recommendation-text {
        font-size: 15px;
        line-height: 1.5;
        color: #344054;
    }
    
    /* =========================================================
       GROQ AI STRATEGY
       ========================================================= */

    .ai-header-card {
        background: #FFFFFF;
        border: 1px solid #E4E7EC;
        border-left: 5px solid #7C3AED;
        border-radius: 12px;
        padding: 18px;
        margin-top: 12px;
        margin-bottom: 15px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.04);
    }

    .ai-title {
        color: #172B4D;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .ai-subtitle {
        color: #667085;
        font-size: 14px;
        line-height: 1.5;
    }
    
    /*  =========================================================
        GROQ AI RESULT
        ========================================================= */

    .ai-result-title {
        color: #172B4D !important;
        font-size: 22px;
        font-weight: 700;
        margin-top: 22px;
        margin-bottom: 12px;
    }

    /* Text inside bordered Streamlit containers */
    [data-testid="stVerticalBlockBorderWrapper"]
    [data-testid="stMarkdownContainer"] {
        color: #172B4D !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"]
    [data-testid="stMarkdownContainer"] p,

    [data-testid="stVerticalBlockBorderWrapper"]
    [data-testid="stMarkdownContainer"] li,

    [data-testid="stVerticalBlockBorderWrapper"]
    [data-testid="stMarkdownContainer"] strong,

    [data-testid="stVerticalBlockBorderWrapper"]
    [data-testid="stMarkdownContainer"] h1,

    [data-testid="stVerticalBlockBorderWrapper"]
    [data-testid="stMarkdownContainer"] h2,

    [data-testid="stVerticalBlockBorderWrapper"]
    [data-testid="stMarkdownContainer"] h3,

    [data-testid="stVerticalBlockBorderWrapper"]
    [data-testid="stMarkdownContainer"] h4 {

        color: #172B4D !important;
    }
    
    /* =========================================================
       FIX ALL MAIN-PAGE MARKDOWN TEXT COLORS
       ========================================================= */

    [data-testid="stMain"] [data-testid="stMarkdownContainer"] {
        color: #172B4D !important;
    }

    [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] em,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] span,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] h5,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] h6 {
        color: #172B4D !important;
    }
    
    /* =========================================================
       GROQ AI RESULT OUTER BORDER
       ========================================================= */

    [data-testid="stMain"]
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 1.5px solid #D0D5DD !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0px 2px 8px rgba(16, 24, 40, 0.06) !important;
    }


    /* =========================================================
    MARKDOWN TABLE BORDERS
    ========================================================= */

    [data-testid="stMain"]
    [data-testid="stMarkdownContainer"] table {
        width: 100% !important;
        border-collapse: collapse !important;
        border: 1px solid #BFC7D5 !important;
        margin-top: 14px !important;
        margin-bottom: 18px !important;
        background-color: #FFFFFF !important;
    }


    /* Table headings */
    [data-testid="stMain"]
    [data-testid="stMarkdownContainer"] table th {
        border: 1px solid #BFC7D5 !important;
        background-color: #EEF2F7 !important;
        color: #172B4D !important;
        font-weight: 700 !important;
        padding: 12px 14px !important;
        text-align: left !important;
    }


    /* Table cells */
    [data-testid="stMain"]
    [data-testid="stMarkdownContainer"] table td {
        border: 1px solid #D0D5DD !important;
        color: #344054 !important;
        padding: 12px 14px !important;
        vertical-align: top !important;
    }


    /* Alternate table rows */
    [data-testid="stMain"]
    [data-testid="stMarkdownContainer"] table tbody tr:nth-child(even) {
        background-color: #F8FAFC !important;
    }


    /* Prevent overflowing long text */
    [data-testid="stMain"]
    [data-testid="stMarkdownContainer"] table th,
    [data-testid="stMain"]
    [data-testid="stMarkdownContainer"] table td {
        white-space: normal !important;
        word-break: normal !important;
        overflow-wrap: anywhere !important;
    }
    
    /* =========================================================
       COMPLETE GROQ RESULT CARD
       ========================================================= */

    .st-key-groq_result_card {
        background-color: #FFFFFF !important;

        border: 2px solid #D0D5DD !important;

        border-radius: 16px !important;

        padding: 28px 32px !important;

        margin-top: 18px !important;
        margin-bottom: 25px !important;

        box-shadow:
            0px 4px 14px rgba(16, 24, 40, 0.08) !important;

        overflow: hidden !important;
    }


    /* Groq card title */

    .st-key-groq_result_card .ai-result-title {
        color: #172B4D !important;
        font-size: 22px !important;
        font-weight: 700 !important;

        padding-bottom: 15px !important;
        margin-bottom: 20px !important;

        border-bottom: 1px solid #E4E7EC !important;
    }


    /* All generated text */

    .st-key-groq_result_card
    [data-testid="stMarkdownContainer"] {

        color: #172B4D !important;
    }


    .st-key-groq_result_card
    [data-testid="stMarkdownContainer"] p,

    .st-key-groq_result_card
    [data-testid="stMarkdownContainer"] li,

    .st-key-groq_result_card
    [data-testid="stMarkdownContainer"] strong,

    .st-key-groq_result_card
    [data-testid="stMarkdownContainer"] h1,

    .st-key-groq_result_card
    [data-testid="stMarkdownContainer"] h2,

    .st-key-groq_result_card
    [data-testid="stMarkdownContainer"] h3,

    .st-key-groq_result_card
    [data-testid="stMarkdownContainer"] h4 {

        color: #172B4D !important;
    }


    /* Footer */

    .ai-result-footer {
        color: #667085 !important;
        font-size: 13px !important;

        border-top: 1px solid #E4E7EC;

        padding-top: 15px;
        margin-top: 20px;
    }
    
    /* =========================================================
       ABOUT PAGE FINAL MODEL CARD
       ========================================================= */

    .about-model-card {
        background-color: #FFFFFF;
        border: 1px solid #E4E7EC;
        border-radius: 12px;
        padding: 18px;
        min-height: 102px;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.04);
    }

    .about-model-label {
        color: #667085;
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 10px;
    }

    .about-model-value {
        color: #172B4D;
        font-size: 20px;
        font-weight: 700;
        line-height: 1.25;
        white-space: normal;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📊 Churn Intelligence")

    st.markdown(
        """
        AI-powered customer churn prediction,
        explainability and retention analytics.
        """
    )

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Select Page",
        [
            "Customer Intelligence",
            "Model Information",
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.caption(
        "ML + SHAP + Rule Engine + Groq LLM"
    )


# ============================================================
# CUSTOMER INTELLIGENCE PAGE
# ============================================================

if page == "Customer Intelligence":

    st.markdown(
        '<div class="main-title">'
        'Customer Churn Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-Powered Customer Risk, Explainability '
        'and Personalized Retention Strategy'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="custom-card">'
        '<b>Customer Analysis Dashboard</b><br>'
        'Select a customer to generate churn prediction, '
        'SHAP explanations and personalized retention actions.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'Customer Selection'
        '</div>',
        unsafe_allow_html=True
    )

    customer_ids = sorted(
        customer_data["customerID"]
        .astype(str)
        .tolist()
    )

    customer_id = st.selectbox(
        "Customer ID",
        ["Select Customer"] + customer_ids
    )

    if customer_id != "Select Customer":

        # ========================================================
        # GET SELECTED CUSTOMER
        # ========================================================

        customer = customer_data[
            customer_data["customerID"] == customer_id
        ].copy()


        # ========================================================
        # PREPARE FEATURES
        # ========================================================

        features = customer.drop(
            columns=[
                "customerID",
                "Churn"
            ]
        )


        # ========================================================
        # MAKE PREDICTION
        # ========================================================

        probability = churn_model.predict_proba(
            features
        )[0][1]

        prediction = churn_model.predict(
            features
        )[0]

        risk_level = get_risk_level(
            probability
        )


        # ========================================================
        # DISPLAY CUSTOMER ID
        # ========================================================

        st.success(
            f"Analyzing Customer: {customer_id}"
        )


        # ========================================================
        # KPI CARDS
        # ========================================================

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )


        with col2:

            st.metric(
                "Risk Level",
                risk_level
            )


        with col3:

            st.metric(
                "Monthly Charges",
                f"${customer['MonthlyCharges'].iloc[0]:.2f}"
            )


        with col4:

            st.metric(
                "Tenure",
                f"{customer['tenure'].iloc[0]} months"
            )


        # ========================================================
        # PREDICTION RESULT
        # ========================================================

        if prediction == 1:

            st.error(
                "Prediction: Customer is likely to churn."
            )

        else:

            st.success(
                "Prediction: Customer is likely to stay."
            )
           
            # ========================================================
            # CUSTOMER PROFILE
            # ========================================================

        st.markdown(
            '<div class="section-title">'
            'Customer Profile'
            '</div>',
            unsafe_allow_html=True
        )

        profile_col1, profile_col2, profile_col3, profile_col4 = (
            st.columns(4)
        )

        with profile_col1:
            st.markdown(
                f"""
                <div class="profile-card">
                    <div class="profile-label">Contract</div>
                    <div class="profile-value">
                        {customer["Contract"].iloc[0]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with profile_col2:
            st.markdown(
                f"""
                <div class="profile-card">
                    <div class="profile-label">Payment Method</div>
                    <div class="profile-value">
                        {customer["PaymentMethod"].iloc[0]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with profile_col3:
            st.markdown(
                f"""
                <div class="profile-card">
                    <div class="profile-label">Internet Service</div>
                    <div class="profile-value">
                        {customer["InternetService"].iloc[0]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with profile_col4:
            st.markdown(
                f"""
                <div class="profile-card">
                    <div class="profile-label">Total Charges</div>
                    <div class="profile-value">
                        ${customer["TotalCharges"].iloc[0]:,.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # ========================================================
        # CHURN RISK GAUGE
        # ========================================================

        st.markdown(
            '<div class="section-title">'
            'Churn Risk Assessment'
            '</div>',
            unsafe_allow_html=True
        )

        gauge_value = probability * 100


        if gauge_value >= 70:

            gauge_color = "#DC2626"

        elif gauge_value >= 40:

            gauge_color = "#F59E0B"

        else:

            gauge_color = "#16A34A"


        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",

                value=gauge_value,

                number={
                    "suffix": "%",
                    "font": {
                        "size": 38
                    }
                },

                title={
                    "text": "Predicted Churn Risk"
                },

                gauge={
                    "axis": {
                        "range": [0, 100]
                    },

                    "bar": {
                        "color": gauge_color
                    },

                    "steps": [
                        {
                            "range": [0, 40],
                            "color": "#DCFCE7"
                        },
                        {
                            "range": [40, 70],
                            "color": "#FEF3C7"
                        },
                        {
                            "range": [70, 100],
                            "color": "#FEE2E2"
                        }
                    ],

                    "threshold": {
                        "line": {
                            "color": "#172B4D",
                            "width": 3
                        },

                        "thickness": 0.8,

                        "value": gauge_value
                    }
                }
            )
        )


        gauge.update_layout(
            height=300,

            margin=dict(
                l=30,
                r=30,
                t=60,
                b=20
            ),

            paper_bgcolor="rgba(0,0,0,0)",

            font={
                "color": "#172B4D"
            }
        )


        st.plotly_chart(
            gauge,
            use_container_width=True
        ) 
        
    # ========================================================
    # SHAP EXPLAINABILITY
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Why Is This Customer at Risk?'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            color:#667085;
            font-size:14px;
            margin-bottom:20px;
        ">
            SHAP explains which model features pushed this
            customer's prediction toward higher or lower churn risk.
            These are model influences, not proven causes.
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # FILTER SHAP DATA FOR SELECTED CUSTOMER
    # --------------------------------------------------------

    customer_shap = shap_data[
        shap_data["customer_id"].astype(str)
        == str(customer_id)
    ].copy()


    if not customer_shap.empty:

        increasing_factors = (
            customer_shap[
                customer_shap["impact_direction"]
                == "INCREASES RISK"
            ]
            .sort_values(
                "absolute_shap_value",
                ascending=True
            )
        )

        reducing_factors = (
            customer_shap[
                customer_shap["impact_direction"]
                == "REDUCES RISK"
            ]
            .sort_values(
                "absolute_shap_value",
                ascending=True
            )
        )


        # Make feature names easier to read
        increasing_factors["display_feature"] = (
            increasing_factors["feature"]
            .astype(str)
            .str.replace("_", " ", regex=False)
        )

        reducing_factors["display_feature"] = (
            reducing_factors["feature"]
            .astype(str)
            .str.replace("_", " ", regex=False)
        )


        # ====================================================
        # TWO SHAP CHARTS
        # ====================================================

        shap_col1, shap_col2 = st.columns(2)


        # ----------------------------------------------------
        # FACTORS INCREASING RISK
        # ----------------------------------------------------

        with shap_col1:

            st.markdown(
                """
                <div style="
                    color:#172B4D;
                    font-size:20px;
                    font-weight:700;
                    margin-bottom:10px;
                ">
                    🔴 Factors Increasing Churn Risk
                </div>
                """,
                unsafe_allow_html=True
            )

            fig_increase = go.Figure()

            fig_increase.add_trace(
                go.Bar(
                    x=increasing_factors[
                        "absolute_shap_value"
                    ],
                    y=increasing_factors[
                        "display_feature"
                    ],
                    orientation="h",
                    marker_color="#DC2626",
                    text=increasing_factors[
                        "shap_value"
                    ].round(3),
                    textposition="auto"
                )
            )

            fig_increase.update_layout(
                height=380,

                margin=dict(
                    l=170,
                    r=20,
                    t=20,
                    b=65
                ),

                xaxis_title="SHAP Impact Magnitude",
                yaxis_title="",

                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",

                font=dict(
                    color="#172B4D"
                )
            )

            fig_increase.update_xaxes(
                tickfont=dict(
                    color="#667085",
                    size=11
                ),
                title_font=dict(
                    color="#667085",
                    size=12
                ),
                title_standoff=15,
                gridcolor="#E4E7EC"
            )

            fig_increase.update_yaxes(
                tickfont=dict(
                    color="#344054",
                    size=11
                ),
                automargin=True
            )
            
            fig_increase.update_xaxes(
                tickfont=dict(
                    color="#667085",
                    size=11
                ),
                title_font=dict(
                    color="#667085",
                    size=12
                ),
                gridcolor="#E4E7EC"
            )

            fig_increase.update_yaxes(
                tickfont=dict(
                    color="#344054",
                    size=11
                )
            )
                               

            st.plotly_chart(
                fig_increase,
                use_container_width=True,
                theme=None
            )


        # ----------------------------------------------------
        # FACTORS REDUCING RISK
        # ----------------------------------------------------

        with shap_col2:

            st.markdown(
                """
                <div style="
                    color:#172B4D;
                    font-size:20px;
                    font-weight:700;
                    margin-bottom:10px;
                ">
                    🟢 Factors Reducing Churn Risk
                </div>
                """,
                unsafe_allow_html=True
            )

            fig_reduce = go.Figure()

            fig_reduce.add_trace(
                go.Bar(
                    x=reducing_factors[
                        "absolute_shap_value"
                    ],
                    y=reducing_factors[
                        "display_feature"
                    ],
                    orientation="h",
                    marker_color="#16A34A",
                    text=reducing_factors[
                        "shap_value"
                    ].round(3),
                    textposition="auto"
                )
            )

            fig_reduce.update_layout(
                height=380,

                margin=dict(
                    l=170,
                    r=20,
                    t=20,
                    b=65
                ),

                xaxis_title="SHAP Impact Magnitude",
                yaxis_title="",

                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",

                font=dict(
                    color="#172B4D"
                )
            )

            fig_reduce.update_xaxes(
                tickfont=dict(
                    color="#667085",
                    size=11
                ),
                title_font=dict(
                    color="#667085",
                    size=12
                ),
                title_standoff=15,
                gridcolor="#E4E7EC"
            )

            fig_reduce.update_yaxes(
                tickfont=dict(
                    color="#344054",
                    size=11
                ),
                automargin=True
            )
            
            fig_reduce.update_xaxes(
                tickfont=dict(
                    color="#667085",
                    size=11
                ),
                title_font=dict(
                    color="#667085",
                    size=12
                ),
                gridcolor="#E4E7EC"
            )

            fig_reduce.update_yaxes(
                tickfont=dict(
                    color="#344054",
                    size=11
                )
            )

            st.plotly_chart(
                fig_reduce,
                use_container_width=True,
                theme=None
            )
        
        # ========================================================
        # RULE-BASED RETENTION RECOMMENDATIONS
        # ========================================================

            st.markdown(
                '<div class="section-title">'
                'Rule-Based Retention Recommendations'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div style="
                    color:#667085;
                    font-size:14px;
                    margin-bottom:18px;
                ">
                    Recommended retention actions generated from
                    predefined business rules using the customer's
                    risk level, profile and model-identified factors.
                </div>
                """,
                unsafe_allow_html=True
            )


            # Filter recommendations for selected customer
            customer_recommendations = (
                recommendations_data[
                    recommendations_data["customer_id"].astype(str)
                    == str(customer_id)
                ]
                .copy()
            )


            # Sort by priority/rank
            if "recommendation_rank" in customer_recommendations.columns:

                customer_recommendations = (
                    customer_recommendations.sort_values(
                        "recommendation_rank"
                    )
                )


            if not customer_recommendations.empty:

                for index, row in customer_recommendations.iterrows():

                    rank = row.get(
                        "recommendation_rank",
                        index + 1
                    )

                    recommendation = str(
                        row["recommendation"]
                    )

                    st.markdown(
                        f"""<div class="recommendation-card">
                            <div class="recommendation-number">PRIORITY {rank}</div>
                            <div class="recommendation-text">{recommendation}</div>
                            </div>""",
                        unsafe_allow_html=True
                    )

            else:

                st.warning(
                    "No retention recommendations were found "
                    "for this customer."
                )
        
        # ========================================================
        # GROQ AI PERSONALIZED RETENTION STRATEGY
        # ========================================================

            st.markdown(
                """<div class="ai-header-card">
            <div class="ai-title">✨ AI Personalized Retention Strategy</div>
            <div class="ai-subtitle">Generate a personalized retention strategy using the customer's churn prediction, SHAP factors, business rules and Groq-hosted Qwen LLM.</div>
            </div>""",
                unsafe_allow_html=True
            ) 
            
        # --------------------------------------------------------
        # PREPARE SHAP FACTORS FOR LLM
        # --------------------------------------------------------

            llm_risk_factors = []

            if not customer_shap.empty:

                llm_increasing = (
                    customer_shap[
                        customer_shap["impact_direction"]
                        == "INCREASES RISK"
                    ]
                    .sort_values(
                        "absolute_shap_value",
                        ascending=False
                    )
                    .head(5)
                )

                for _, shap_row in llm_increasing.iterrows():

                    llm_risk_factors.append(
                        {
                            "feature": shap_row["feature"],
                            "shap_value": round(
                                float(shap_row["shap_value"]),
                                4
                            )
                        }
         
                    )          
        
        # --------------------------------------------------------
        # PREPARE RULE RECOMMENDATIONS FOR LLM
        # --------------------------------------------------------

            llm_rule_recommendations = []

            if not customer_recommendations.empty:

                llm_rule_recommendations = (
                    customer_recommendations[
                        "recommendation"
                    ]
                    .astype(str)
                    .tolist()
                )
        
        # --------------------------------------------------------
        # SESSION STATE KEY
        # --------------------------------------------------------

            strategy_key = (
                f"llm_strategy_{customer_id}"
            )


            # --------------------------------------------------------
            # GENERATE BUTTON
            # --------------------------------------------------------

            generate_ai = st.button(
                "✨ Generate AI Retention Strategy",
                type="primary",
                use_container_width=True
            )
            
            if generate_ai:

                with st.spinner(
                    "Groq AI is analyzing this customer..."
                ):

                    llm_strategy = generate_llm_strategy(
                        customer=customer,
                        churn_probability=probability,
                        risk_level=risk_level,
                        risk_factors=llm_risk_factors,
                        rule_recommendations=llm_rule_recommendations
                    )

                    st.session_state[
                        strategy_key
                    ] = llm_strategy
        
        # --------------------------------------------------------
        # DISPLAY SAVED AI STRATEGY
        # --------------------------------------------------------

            if strategy_key in st.session_state:

                strategy_text = st.session_state[
                    strategy_key
                ]

                with st.container(
                    key="groq_result_card"
                ):

                    st.markdown(
                        '<div class="ai-result-title">'
                        '🤖 Groq AI Recommendation'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        strategy_text
                    )

                    st.markdown(
                        """
                        <div class="ai-result-footer">
                            Generated using Groq + Qwen from the ML prediction,
                            SHAP explanation and rule-based recommendations.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
               
    else:

        st.warning(
            "SHAP explanation was not found "
            "for this customer."
        )   

# ============================================================
# MODEL INFORMATION PAGE
# ============================================================

elif page == "Model Information":

    st.markdown(
        '<div class="main-title">'
        'Machine Learning Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Evaluation and comparison of churn prediction models'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # MODEL METRICS
    # ========================================================

    logistic_row = model_results[
        model_results["Model"] == "Logistic Regression"
    ].iloc[0]

    random_forest_row = model_results[
        model_results["Model"] == "Random Forest"
    ].iloc[0]


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Best ROC-AUC",
            f"{logistic_row['ROC-AUC'] * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Best Recall",
            f"{logistic_row['Recall'] * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Best F1 Score",
            f"{random_forest_row['F1 Score'] * 100:.2f}%"
        )

    with col4:
        st.metric(
            "Best Accuracy",
            f"{random_forest_row['Accuracy'] * 100:.2f}%"
        )


    # ========================================================
    # SELECTED MODEL
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Selected Final Model'
        '</div>',
        unsafe_allow_html=True
    )

    st.success(
        "Logistic Regression"
    )

    st.markdown(
        """
        Logistic Regression was selected as the final churn
        prediction model because it achieved the highest
        ROC-AUC and Recall.

        **ROC-AUC** measures how well the model ranks churners
        above non-churners.

        **Recall** is particularly important in churn analysis
        because it measures how many actual churners are
        successfully identified.
        """
    )


    # ========================================================
    # MODEL COMPARISON CHART
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Model Performance Comparison'
        '</div>',
        unsafe_allow_html=True
    )

    comparison_long = model_results.melt(
        id_vars="Model",
        value_vars=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],
        var_name="Metric",
        value_name="Score"
    )


    comparison_fig = go.Figure()


    for model_name in comparison_long["Model"].unique():

        model_data = comparison_long[
            comparison_long["Model"] == model_name
        ]

        comparison_fig.add_trace(
            go.Bar(
                name=model_name,
                x=model_data["Metric"],
                y=model_data["Score"] * 100,
                text=(
                    model_data["Score"] * 100
                ).round(2),
                texttemplate="%{text:.2f}%",
                textposition="outside"
            )
        )


    comparison_fig.update_layout(
        barmode="group",
        height=450,
        yaxis_title="Score (%)",
        xaxis_title="",
        yaxis=dict(
            range=[0, 100]
        ),
        legend_title="Model",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#172B4D"
        ),
        margin=dict(
            l=40,
            r=20,
            t=30,
            b=40
        )
    )


    st.plotly_chart(
        comparison_fig,
        use_container_width=True,
        theme=None
    )
    
    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Confusion Matrix — Test Set'
        '</div>',
        unsafe_allow_html=True
    )


    # Calculate confusion matrix values
    true_positive = len(
        model_test_data[
            (model_test_data["actual_churn"] == 1)
            &
            (model_test_data["predicted_churn"] == 1)
        ]
    )

    false_negative = len(
        model_test_data[
            (model_test_data["actual_churn"] == 1)
            &
            (model_test_data["predicted_churn"] == 0)
        ]
    )

    false_positive = len(
        model_test_data[
            (model_test_data["actual_churn"] == 0)
            &
            (model_test_data["predicted_churn"] == 1)
        ]
    )

    true_negative = len(
        model_test_data[
            (model_test_data["actual_churn"] == 0)
            &
            (model_test_data["predicted_churn"] == 0)
        ]
    )
    
    cm1, cm2, cm3, cm4 = st.columns(4)

    with cm1:
        st.metric(
            "True Positives",
            true_positive
        )

    with cm2:
        st.metric(
            "False Negatives",
            false_negative
        )

    with cm3:
        st.metric(
            "False Positives",
            false_positive
        )

    with cm4:
        st.metric(
            "True Negatives",
            true_negative
        )
    
    confusion_values = [
        [
            true_positive,
            false_negative
        ],
        [
            false_positive,
            true_negative
        ]
    ]


    confusion_fig = go.Figure(
        data=go.Heatmap(

            # Semantic color codes:
            # 2 = correct prediction
            # 1 = false positive
            # 0 = false negative
            z=[
                [2, 0],
                [1, 2]
            ],

            x=[
                "Predicted Churn",
                "Predicted Stay"
            ],

            y=[
                "Actual Churn",
                "Actual Stay"
            ],

            text=[
                [
                    f"TP<br>{true_positive}",
                    f"FN<br>{false_negative}"
                ],
                [
                    f"FP<br>{false_positive}",
                    f"TN<br>{true_negative}"
                ]
            ],

            texttemplate="%{text}",

            textfont={
                "size": 18,
                "color": "#172B4D"
            },

            colorscale=[
                [0.0000, "#FEE2E2"],
                [0.3333, "#FEE2E2"],

                [0.3334, "#FEF3C7"],
                [0.6666, "#FEF3C7"],

                [0.6667, "#DCFCE7"],
                [1.0000, "#DCFCE7"]
            ],

            zmin=0,
            zmax=2,
            showscale=False
        )
    )


    confusion_fig.update_layout(
        height=380,

        xaxis_title="Predicted Outcome",
        yaxis_title="Actual Outcome",

        paper_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="#172B4D"
        ),

        margin=dict(
            l=80,
            r=30,
            t=30,
            b=60
        )
    )

    confusion_fig.update_yaxes(
        autorange="reversed"
    )


    st.plotly_chart(
        confusion_fig,
        use_container_width=True,
        theme=None
    ) 
    
    st.markdown(
        '<div class="section-title">'
        'Business Interpretation'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        """
        The Logistic Regression model correctly identified
        298 customers who actually churned and 740 customers
        who stayed.

        It missed 76 actual churners, while 295 customers were
        flagged as churn risks even though they stayed.

        Because customer-retention problems place high importance
        on identifying potential churners early, the model's
        79.68% recall and 84.27% ROC-AUC make it suitable for
        prioritizing customers for proactive retention actions.
        """
    )
    
# ============================================================
# ABOUT PROJECT PAGE
# ============================================================

elif page == "About Project":

    # ========================================================
    # PAGE HEADER
    # ========================================================

    st.markdown(
        '<div class="main-title">'
        'About the Project'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-Powered Customer Churn Intelligence System'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # PROJECT OVERVIEW
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Project Overview'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        This end-to-end **Customer Churn Intelligence System**
        predicts the probability that a telecom customer will
        churn, explains the major factors influencing that
        prediction, and generates actionable retention
        strategies.

        The project combines **Machine Learning, Explainable AI,
        SQL Analytics, Power BI, Generative AI and Streamlit**
        into one complete business intelligence solution.
        """
    )


    # ========================================================
    # PROJECT KPIs
    # ========================================================

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric(
            "Customers Analyzed",
            "7,043"
        )

    with kpi2:
        st.markdown(
            '<div class="about-model-card">'
            '<div class="about-model-label">Final Model</div>'
            '<div class="about-model-value">Logistic Regression</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with kpi3:
        st.metric(
            "ROC-AUC",
            "84.27%"
        )

    with kpi4:
        st.metric(
            "Recall",
            "79.68%"
        )


    # ========================================================
    # BUSINESS PROBLEM
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Business Problem'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        Customer churn directly affects recurring revenue and
        customer lifetime value.

        Traditional churn models often provide only a prediction.
        This project goes further by answering three important
        business questions:

        **1. Who is likely to churn?**

        **2. Why is the customer considered at risk?**

        **3. What retention action should the business take?**
        """
    )


    # ========================================================
    # END-TO-END WORKFLOW
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'End-to-End Intelligence Workflow'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            background:#FFFFFF;
            border:1px solid #E4E7EC;
            border-radius:12px;
            padding:22px;
            color:#172B4D;
            line-height:2;
            text-align:center;
            font-weight:600;
        ">
            Customer Data
            &nbsp; → &nbsp;
            Data Cleaning
            &nbsp; → &nbsp;
            Feature Engineering
            &nbsp; → &nbsp;
            Machine Learning
            &nbsp; → &nbsp;
            Churn Probability
            <br><br>
            SHAP Explainability
            &nbsp; → &nbsp;
            Rule-Based Retention Engine
            &nbsp; → &nbsp;
            Groq + Qwen LLM
            &nbsp; → &nbsp;
            Personalized Retention Strategy
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # TECHNOLOGY STACK
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Technology Stack'
        '</div>',
        unsafe_allow_html=True
    )

    tech1, tech2, tech3 = st.columns(3)

    with tech1:

        st.markdown(
            """
            ### Data & Machine Learning

            - Python
            - Pandas
            - NumPy
            - Scikit-learn
            - Logistic Regression
            - Random Forest
            - Joblib
            """
        )

    with tech2:

        st.markdown(
            """
            ### Analytics & Explainability

            - SHAP
            - SQLite
            - SQL
            - Power BI
            - Plotly
            - Exploratory Data Analysis
            """
        )

    with tech3:

        st.markdown(
            """
            ### AI & Application

            - Groq API
            - Qwen LLM
            - Prompt Engineering
            - Streamlit
            - Python-dotenv
            - Generative AI
            """
        )


    # ========================================================
    # SYSTEM CAPABILITIES
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'System Capabilities'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        - Predicts customer churn probability in real time.
        - Classifies customers into LOW, MEDIUM and HIGH risk.
        - Explains individual predictions using SHAP.
        - Identifies factors increasing and reducing churn risk.
        - Generates deterministic rule-based retention actions.
        - Uses Groq-hosted Qwen to generate personalized
          retention strategies.
        - Provides model evaluation and confusion-matrix analysis.
        - Supports business reporting through Power BI.
        - Presents all intelligence through an interactive
          Streamlit application.
        """
    )


    # ========================================================
    # MODEL SELECTION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Machine Learning Model'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        """
        Logistic Regression was selected as the final model
        because it achieved the highest ROC-AUC of 84.27%
        and Recall of 79.68%.

        For a customer-retention use case, Recall is especially
        important because missing an actual churner may lead to
        lost customers and recurring revenue.
        """
    )


    # ========================================================
    # RESPONSIBLE AI
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Explainable & Responsible AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        SHAP values are used to explain how model features
        influence each churn prediction.

        These values represent **model influence rather than
        proven causal relationships**.

        The Generative AI layer is also instructed to use only
        supplied customer data, model outputs, SHAP explanations
        and business rules instead of inventing unsupported
        customer information.
        """
    )


    # ========================================================
    # PORTFOLIO VALUE
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Skills Demonstrated'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        **Data Analysis • SQL • EDA • Feature Engineering •
        Machine Learning • Model Evaluation • Explainable AI •
        SHAP • Power BI • Generative AI • API Integration •
        Prompt Engineering • Streamlit • Business Storytelling**
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
    Customer Churn Intelligence System |
    Machine Learning • SHAP • Groq AI • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)