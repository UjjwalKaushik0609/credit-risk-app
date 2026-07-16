import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import io

# ============================================================================
# BACKEND — DO NOT MODIFY
# (model loading, prediction logic, and column mappings are unchanged)
# ============================================================================

COLUMN_NAMES = {
    # Existing mapped columns
    'NETMONTHLYINCOME': 'Net Monthly Income',
    'Time_With_Curr_Empr': 'Employment Tenure',
    'CC_Flag': 'Credit Card Holder',
    'PL_Flag': 'Personal Loan Holder',
    'HL_Flag': 'Home Loan Holder',
    'GL_Flag': 'Gold Loan Holder',
    'EDUCATION': 'Education Level',
    'MARITALSTATUS_Married': 'Married Status',
    'MARITALSTATUS_Single': 'Single Status',
    'GENDER_F': 'Female',
    'GENDER_M': 'Male',

    # Additional columns to map
    'pct_tl_open_L6M': 'Percent Total Lines Open (Last 6M)',
    'pct_tl_closed_L6M': 'Percent Total Lines Closed (Last 6M)',
    'Tot_TL_closed_L12M': 'Total Lines Closed (Last 12M)',
    'pct_tl_closed_L12M': 'Percent Total Lines Closed (Last 12M)',
    'Tot_Missed_Pmnt': 'Total Missed Payments',
    'CC_TL': 'Credit Card Total Lines',
    'Home_TL': 'Home Loan Total Lines',
    'PL_TL': 'Personal Loan Total Lines',
    'Secured_TL': 'Secured Loan Total Lines',
    'Unsecured_TL': 'Unsecured Loan Total Lines',
    'Other_TL': 'Other Loan Total Lines',
    'Age_Oldest_TL': 'Age of Oldest Trade Line',
    'Age_Newest_TL': 'Age of Newest Trade Line',
    'time_since_recent_payment': 'Time Since Recent Payment',
    'max_recent_level_of_deliq': 'Max Recent Delinquency Level',
    'num_deliq_6_12mts': 'Delinquencies in 6-12 Months',
    'num_times_60p_dpd': 'Times 60+ Days Past Due',
    'num_std_12mts': 'Standard Accounts in 12 Months',

    # Additional columns
    'num_sub': 'Number of Substandard Accounts',
    'num_sub_6mts': 'Number of Substandard Accounts (6 Months)',
    'num_sub_12mts': 'Number of Substandard Accounts (12 Months)',
    'num_dbt': 'Number of Doubtful Accounts',
    'num_dbt_12mts': 'Number of Doubtful Accounts (12 Months)',
    'num_lss': 'Number of Loss Accounts',
    'recent_level_of_deliq': 'Recent Delinquency Level',
    'CC_enq_L12m': 'Credit Card Inquiries (Last 12M)',
    'PL_enq_L12m': 'Personal Loan Inquiries (Last 12M)',
    'time_since_recent_enq': 'Time Since Recent Inquiry',
    'enq_L3m': 'Inquiries in Last 3 Months',
    'pct_PL_enq_L6m_of_ever': 'Percent Personal Loan Inquiries (Last 6M)',
    'pct_CC_enq_L6m_of_ever': 'Percent Credit Card Inquiries (Last 6M)',

    # Product Inquiry Columns
    'last_prod_enq2_AL': 'Last Product Inquiry - Auto Loan',
    'last_prod_enq2_CC': 'Last Product Inquiry - Credit Card',
    'last_prod_enq2_ConsumerLoan': 'Last Product Inquiry - Consumer Loan',
    'last_prod_enq2_HL': 'Last Product Inquiry - Home Loan',
    'last_prod_enq2_PL': 'Last Product Inquiry - Personal Loan',
    'last_prod_enq2_others': 'Last Product Inquiry - Others',

    # First Product Inquiry Columns
    'first_prod_enq2_AL': 'First Product Inquiry - Auto Loan',
    'first_prod_enq2_CC': 'First Product Inquiry - Credit Card',
    'first_prod_enq2_ConsumerLoan': 'First Product Inquiry - Consumer Loan',
    'first_prod_enq2_HL': 'First Product Inquiry - Home Loan',
    'first_prod_enq2_PL': 'First Product Inquiry - Personal Loan',
    'first_prod_enq2_others': 'First Product Inquiry - Others'
}


def load_model():
    with open('xgb_classifier.pkl', 'rb') as f:
        return pickle.load(f)


def predict_credit_approval(model, input_data):
    probabilities = model.predict_proba(input_data)
    return probabilities


# ============================================================================
# FRONTEND — THEME, LAYOUT & UX (redesign only, no backend changes)
# ============================================================================

PRIMARY = "#2563EB"
SUCCESS = "#16A34A"
WARNING = "#F59E0B"
DANGER = "#DC2626"
BG = "#0E1117"
SIDEBAR_BG = "#161B22"

RISK_LABELS = [
    "P1 · Low Risk",
    "P2 · Moderate Risk",
    "P3 · High Risk",
    "P4 · Very High Risk",
]
RISK_COLORS = [SUCCESS, PRIMARY, WARNING, DANGER]


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, sans-serif;
    }}

    .stApp {{
        background: radial-gradient(circle at top left, #131826 0%, {BG} 45%, #0a0d12 100%);
        color: #E6E8EC;
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {SIDEBAR_BG} 0%, #10141a 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }}

    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stSlider label {{
        color: #AEB4C0 !important;
        font-size: 0.85rem;
        font-weight: 500;
    }}

    /* Header banner */
    .dash-header {{
        background: linear-gradient(120deg, rgba(37,99,235,0.18) 0%, rgba(22,27,34,0.4) 100%);
        border: 1px solid rgba(37,99,235,0.25);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 26px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }}
    .dash-header h1 {{
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0;
        color: #F5F7FA;
        letter-spacing: -0.02em;
    }}
    .dash-header p {{
        margin: 6px 0 0 0;
        color: #9AA4B2;
        font-size: 0.95rem;
    }}
    .dash-header .badge {{
        display: inline-block;
        margin-top: 12px;
        padding: 5px 14px;
        border-radius: 999px;
        background: rgba(37,99,235,0.15);
        border: 1px solid rgba(37,99,235,0.4);
        color: #7EA8FF;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}

    /* Glass card */
    .glass-card {{
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 22px 24px;
        backdrop-filter: blur(14px);
        box-shadow: 0 6px 24px rgba(0,0,0,0.3);
        margin-bottom: 18px;
    }}

    /* KPI cards */
    .kpi-card {{
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 16px;
        padding: 20px;
        text-align: left;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 18px rgba(0,0,0,0.3);
        height: 100%;
    }}
    .kpi-label {{
        color: #8B93A3;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 1.7rem;
        font-weight: 800;
        color: #F5F7FA;
    }}
    .kpi-sub {{
        font-size: 0.78rem;
        color: #6E7688;
        margin-top: 4px;
    }}

    /* Recommendation card */
    .rec-card-good {{
        background: rgba(22,163,74,0.10);
        border: 1px solid rgba(22,163,74,0.35);
        border-radius: 16px;
        padding: 22px 26px;
        margin-top: 6px;
    }}
    .rec-card-bad {{
        background: rgba(220,38,38,0.10);
        border: 1px solid rgba(220,38,38,0.35);
        border-radius: 16px;
        padding: 22px 26px;
        margin-top: 6px;
    }}
    .rec-title {{
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 10px;
    }}
    .rec-line {{
        font-size: 0.92rem;
        margin: 4px 0;
        color: #D6DAE2;
    }}

    /* Buttons */
    div.stButton > button {{
        background: linear-gradient(135deg, {PRIMARY} 0%, #1D4ED8 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1rem;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 6px 18px rgba(37,99,235,0.35);
        transition: transform 0.12s ease;
    }}
    div.stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 8px 22px rgba(37,99,235,0.5);
    }}

    div[data-testid="stDownloadButton"] > button {{
        background: rgba(255,255,255,0.06);
        color: #E6E8EC;
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 12px;
        font-weight: 600;
    }}

    section[data-testid="stSidebar"] .streamlit-expanderHeader {{
        font-weight: 700;
        color: #E6E8EC;
        font-size: 0.92rem;
    }}

    hr {{
        border-color: rgba(255,255,255,0.08);
    }}

    .section-title {{
        font-size: 1.1rem;
        font-weight: 700;
        color: #F5F7FA;
        margin: 6px 0 14px 0;
    }}
    </style>
    """, unsafe_allow_html=True)


def render_header():
    st.markdown(f"""
    <div class="dash-header">
        <h1>💳 Credit Approval Dashboard</h1>
        <p>AI Powered Credit Risk Assessment</p>
        <span class="badge">XGBoost Classification Model</span>
    </div>
    """, unsafe_allow_html=True)


def kpi_card(label, value, sub=""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Frontend-only helper: groups model features into UX sections.
# This purely controls where a field is displayed — it does not alter the
# feature values, names, or the order used at prediction time.
# ----------------------------------------------------------------------------
def categorize_column(col):
    personal = {'NETMONTHLYINCOME', 'Time_With_Curr_Empr', 'EDUCATION',
                'GENDER_F', 'GENDER_M', 'MARITALSTATUS_Married', 'MARITALSTATUS_Single'}
    loans = {'CC_Flag', 'PL_Flag', 'HL_Flag', 'GL_Flag'}
    behaviour = {'pct_tl_open_L6M', 'pct_tl_closed_L6M', 'Tot_TL_closed_L12M', 'pct_tl_closed_L12M',
                 'Tot_Missed_Pmnt', 'Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment',
                 'max_recent_level_of_deliq', 'num_deliq_6_12mts', 'num_times_60p_dpd', 'recent_level_of_deliq'}
    enquiries = {'CC_enq_L12m', 'PL_enq_L12m', 'time_since_recent_enq', 'enq_L3m',
                 'pct_PL_enq_L6m_of_ever', 'pct_CC_enq_L6m_of_ever'}
    product_prefixes = ('last_prod_enq2_', 'first_prod_enq2_')

    if col in personal:
        return 'personal'
    if col in loans:
        return 'loans'
    if col in enquiries:
        return 'enquiries'
    if col.startswith(product_prefixes):
        return 'product_enq'
    if col in behaviour:
        return 'behaviour'
    return 'credit_profile'


def render_sidebar_inputs(model):
    """Builds grouped, human-friendly sidebar inputs.
    Populates the same raw input_data dict/keys the model expects —
    only the UI presentation is redesigned."""
    input_data = {}
    renamed_input_data = {}
    all_cols = list(model.feature_names_in_)
    handled = set()

    def add(col, value, label=None):
        input_data[col] = value
        renamed_input_data[label or COLUMN_NAMES.get(col, col)] = value
        handled.add(col)

    grouped = {'personal': [], 'loans': [], 'credit_profile': [], 'behaviour': [],
               'enquiries': [], 'product_enq': []}
    for c in all_cols:
        grouped[categorize_column(c)].append(c)

    st.sidebar.markdown("### 🏦 Customer Information")

    # ---------------- Personal Details ----------------
    with st.sidebar.expander("👤 Personal Details", expanded=True):
        if 'GENDER_F' in all_cols or 'GENDER_M' in all_cols:
            gender = st.selectbox("Gender", ["Male", "Female"], key="gender_select")
            if 'GENDER_F' in all_cols:
                add('GENDER_F', 1 if gender == "Female" else 0, "Gender")
            if 'GENDER_M' in all_cols and 'GENDER_F' not in all_cols:
                add('GENDER_M', 1 if gender == "Male" else 0, "Gender")
            elif 'GENDER_M' in all_cols:
                add('GENDER_M', 1 if gender == "Male" else 0)

        if 'MARITALSTATUS_Married' in all_cols or 'MARITALSTATUS_Single' in all_cols:
            marital = st.selectbox("Marital Status", ["Single", "Married"], key="marital_select")
            if 'MARITALSTATUS_Married' in all_cols:
                add('MARITALSTATUS_Married', 1 if marital == "Married" else 0, "Marital Status")
            if 'MARITALSTATUS_Single' in all_cols:
                add('MARITALSTATUS_Single', 1 if marital == "Single" else 0,
                    "Marital Status" if 'MARITALSTATUS_Married' not in all_cols else "Single Status (flag)")

        if 'EDUCATION' in all_cols:
            edu = st.selectbox("Education", ["High School", "Diploma", "Graduate", "Post Graduate"], key="edu_select")
            edu_map = {"High School": 0, "Diploma": 1, "Graduate": 2, "Post Graduate": 3}
            add('EDUCATION', edu_map[edu], "Education Level")

        if 'NETMONTHLYINCOME' in all_cols:
            val = st.number_input("Net Monthly Income (₹)", min_value=0.0, value=50000.0, step=1000.0,
                                   key='NETMONTHLYINCOME')
            add('NETMONTHLYINCOME', val)

        if 'Time_With_Curr_Empr' in all_cols:
            val = st.slider("Years with Current Employer", 0, 40, 2, key='Time_With_Curr_Empr')
            add('Time_With_Curr_Empr', val)

        # any other personal-bucket columns not explicitly handled above
        for col in grouped['personal']:
            if col not in handled:
                renamed = COLUMN_NAMES.get(col, col)
                val = st.number_input(renamed, value=0.0, key=col)
                add(col, val)

    # ---------------- Existing Loans ----------------
    with st.sidebar.expander("💳 Existing Loans"):
        loan_labels = {'CC_Flag': 'Credit Card', 'PL_Flag': 'Personal Loan',
                       'HL_Flag': 'Home Loan', 'GL_Flag': 'Gold Loan'}
        for col in ['CC_Flag', 'PL_Flag', 'HL_Flag', 'GL_Flag']:
            if col in all_cols:
                choice = st.selectbox(loan_labels[col], ["No", "Yes"], key=col)
                add(col, 1 if choice == "Yes" else 0, loan_labels[col])
        for col in grouped['loans']:
            if col not in handled:
                renamed = COLUMN_NAMES.get(col, col)
                choice = st.selectbox(renamed, ["No", "Yes"], key=col)
                add(col, 1 if choice == "Yes" else 0)

    # ---------------- Credit Profile ----------------
    with st.sidebar.expander("📊 Credit Profile"):
        for col in grouped['credit_profile']:
            if col in handled:
                continue
            renamed = COLUMN_NAMES.get(col, col)
            val = st.number_input(renamed, value=0.0, step=1.0, key=col)
            add(col, val)

    # ---------------- Credit Behaviour ----------------
    with st.sidebar.expander("📈 Credit Behaviour"):
        pct_cols = {'pct_tl_open_L6M', 'pct_tl_closed_L6M', 'pct_tl_closed_L12M'}
        for col in grouped['behaviour']:
            if col in handled:
                continue
            renamed = COLUMN_NAMES.get(col, col)
            if col in pct_cols:
                val = st.slider(renamed, 0.0, 100.0, 0.0, key=col)
            else:
                val = st.number_input(renamed, value=0.0, step=1.0, key=col)
            add(col, val)

    # ---------------- Credit Enquiries ----------------
    with st.sidebar.expander("🔍 Credit Enquiries"):
        pct_cols = {'pct_PL_enq_L6m_of_ever', 'pct_CC_enq_L6m_of_ever'}
        for col in grouped['enquiries']:
            if col in handled:
                continue
            renamed = COLUMN_NAMES.get(col, col)
            if col in pct_cols:
                val = st.slider(renamed, 0.0, 100.0, 0.0, key=col)
            else:
                val = st.number_input(renamed, value=0.0, step=1.0, key=col)
            add(col, val)

    # ---------------- Product Enquiries ----------------
    with st.sidebar.expander("🏦 Product Enquiries"):
        for col in grouped['product_enq']:
            if col in handled:
                continue
            renamed = COLUMN_NAMES.get(col, col)
            choice = st.selectbox(renamed, ["No", "Yes"], key=col)
            add(col, 1 if choice == "Yes" else 0)

    # Defensive catch-all for any unforeseen columns
    leftover = [c for c in all_cols if c not in handled]
    if leftover:
        with st.sidebar.expander("Additional Details"):
            for col in leftover:
                renamed = COLUMN_NAMES.get(col, col)
                val = st.number_input(renamed, value=0.0, key=col)
                add(col, val)

    return input_data, renamed_input_data


# ----------------------------------------------------------------------------
# Result visuals
# ----------------------------------------------------------------------------
def plotly_dark_layout(fig, height=280):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6E8EC", family="Inter"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=height,
    )
    return fig


def render_gauge(risk_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={'suffix': "", 'font': {'size': 34, 'color': '#F5F7FA'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#8B93A3'},
            'bar': {'color': PRIMARY},
            'bgcolor': 'rgba(255,255,255,0.04)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 25], 'color': 'rgba(22,163,74,0.35)'},
                {'range': [25, 50], 'color': 'rgba(37,99,235,0.35)'},
                {'range': [50, 75], 'color': 'rgba(245,158,11,0.35)'},
                {'range': [75, 100], 'color': 'rgba(220,38,38,0.35)'},
            ],
            'threshold': {
                'line': {'color': '#F5F7FA', 'width': 3},
                'thickness': 0.8,
                'value': risk_score
            }
        },
        title={'text': "Risk Score", 'font': {'size': 14, 'color': '#8B93A3'}}
    ))
    return plotly_dark_layout(fig, height=280)


def render_donut(probabilities):
    fig = go.Figure(go.Pie(
        labels=RISK_LABELS,
        values=probabilities,
        hole=0.62,
        marker=dict(colors=RISK_COLORS, line=dict(color=BG, width=2)),
        textinfo="percent",
        textfont=dict(color="#0E1117", size=12, family="Inter")
    ))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.15, font=dict(size=10)))
    return plotly_dark_layout(fig, height=300)


def render_hbar(probabilities):
    fig = go.Figure(go.Bar(
        x=[p * 100 for p in probabilities],
        y=RISK_LABELS,
        orientation='h',
        marker=dict(color=RISK_COLORS),
        text=[f"{p*100:.1f}%" for p in probabilities],
        textposition='outside',
    ))
    fig.update_xaxes(range=[0, 100], showgrid=False, title="Probability (%)")
    fig.update_yaxes(showgrid=False)
    return plotly_dark_layout(fig, height=260)


def render_results(probabilities, renamed_input_data, raw_input_data):
    predicted_class = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    approval_probability = float(probabilities[0] + probabilities[1])
    risk_score = float(
        probabilities[0] * 10 + probabilities[1] * 40 + probabilities[2] * 70 + probabilities[3] * 100
    )
    risk_level_text = ["Low", "Moderate", "High", "Very High"][predicted_class]

    st.markdown('<div class="section-title">📌 Prediction Results</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Approval Probability", f"{approval_probability*100:.1f}%", "P1 + P2 tiers")
    with c2:
        kpi_card("Risk Level", risk_level_text, RISK_LABELS[predicted_class])
    with c3:
        kpi_card("Confidence", f"{confidence*100:.1f}%", "Model certainty")
    with c4:
        kpi_card("Model", "XGBoost", "Classifier")

    st.write("")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(render_gauge(risk_score), use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
    with g2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.plotly_chart(render_donut(probabilities), use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.plotly_chart(render_hbar(probabilities), use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

    # Recommendation card
    missed_payments = next((v for k, v in renamed_input_data.items() if 'Missed Payment' in k), 0)
    recent_enquiries = next((v for k, v in renamed_input_data.items() if 'Inquiries in Last 3 Months' in k), 0)

    if predicted_class <= 1:
        st.markdown(f"""
        <div class="rec-card-good">
            <div class="rec-title" style="color:{SUCCESS};">✓ Recommended for Approval</div>
            <div class="rec-line">✓ Low Credit Risk ({risk_level_text} risk tier)</div>
            <div class="rec-line">✓ Good Payment History</div>
            <div class="rec-line">✓ Eligible for Approval</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="rec-card-bad">
            <div class="rec-title" style="color:{DANGER};">⚠ Manual Verification Required</div>
            <div class="rec-line">⚠ Elevated Credit Risk ({risk_level_text} risk tier)</div>
            <div class="rec-line">⚠ Total Missed Payments on file: {missed_payments}</div>
            <div class="rec-line">⚠ Recent Enquiries (last 3 months): {recent_enquiries}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section-title">🧾 Applicant Summary</div>', unsafe_allow_html=True)

    summary_rows = []
    for col, val in raw_input_data.items():
        summary_rows.append({
            "Category": categorize_column(col).replace('_', ' ').title(),
            "Field": COLUMN_NAMES.get(col, col),
            "Value": val
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(["Category", "Field"]).reset_index(drop=True)
    st.dataframe(summary_df, use_container_width=True, height=360)

    # Export report
    export_buffer = io.StringIO()
    export_df = summary_df.copy()
    export_df.loc[len(export_df)] = ["Prediction", "Risk Level", risk_level_text]
    export_df.loc[len(export_df)] = ["Prediction", "Approval Probability", f"{approval_probability*100:.2f}%"]
    export_df.loc[len(export_df)] = ["Prediction", "Confidence", f"{confidence*100:.2f}%"]
    export_df.to_csv(export_buffer, index=False)

    st.download_button(
        label="📤 Export Report",
        data=export_buffer.getvalue(),
        file_name=f"credit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_placeholder():
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:60px 20px;">
        <div style="font-size:2.2rem;">📋</div>
        <div style="font-size:1.1rem; font-weight:600; color:#D6DAE2; margin-top:10px;">
            Fill in the customer details in the sidebar
        </div>
        <div style="color:#8B93A3; margin-top:6px;">
            Then click <b>Analyze Credit Risk</b> to generate the AI-powered assessment.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN
# ============================================================================
def main():
    st.set_page_config(
        page_title="Credit Approval Dashboard",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_css()
    render_header()

    model = load_model()

    input_data, renamed_input_data = render_sidebar_inputs(model)

    st.sidebar.markdown("---")
    analyze_clicked = st.sidebar.button("🚀 Analyze Credit Risk", use_container_width=True, type="primary")

    if analyze_clicked:
        # Reindex to the model's expected column order before prediction.
        # This does not change prediction logic, feature names, or values —
        # it only guarantees the DataFrame passed to predict_proba matches
        # model.feature_names_in_.
        input_df = pd.DataFrame([input_data])[list(model.feature_names_in_)]
        probabilities = predict_credit_approval(model, input_df)[0]

        st.session_state['probabilities'] = probabilities
        st.session_state['renamed_input_data'] = renamed_input_data
        st.session_state['raw_input_data'] = input_data
        st.session_state['analyzed'] = True

    if st.session_state.get('analyzed'):
        render_results(
            st.session_state['probabilities'],
            st.session_state['renamed_input_data'],
            st.session_state['raw_input_data'],
        )
    else:
        render_placeholder()


if __name__ == "__main__":
    main()
