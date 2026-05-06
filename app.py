import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# 1. MODEL TRAINING (Cached to load instantly)

@st.cache_resource
def train_model():
    df = pd.read_csv("cs-training-clean.csv")
    target_col = [col for col in df.columns if 'dlqin2yrs' in col.lower() or 'serious' in col.lower()][0]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    
    model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric="logloss", use_label_encoder=False)
    model.fit(X_train_scaled, y_train_bal)
    
    return model, scaler, X.columns.tolist()

model, scaler, feature_names = train_model()

# 2. PROFESSIONAL USER INTERFACE

st.set_page_config(page_title="Credit Scoring AI", page_icon="🏦", layout="wide")

# --- HEADER ---
st.title("🏦 AI Credit Risk Assessment System")
st.caption("Predicts the probability of serious delinquency within 24 months based on historical financial data.")
st.markdown("---")

# Create 3 main columns for the form
col_personal, col_credit, col_debt = st.columns(3)

with col_personal:
    st.subheader("👤 Applicant Profile")
    age = st.number_input("Applicant Age", min_value=18, max_value=100, value=35, step=1)
    monthly_income = st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0, step=500.0)
    num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0, step=1)
    st.caption("*(Dependents increase fixed living expenses)*")

with col_credit:
    st.subheader("💳 Credit History")
    st.info("⚠️ Late payments are the strongest predictors of default.")
    past_due_30_59 = st.number_input("Times 30-59 Days Past Due", min_value=0, max_value=20, value=0, step=1)
    past_due_60_89 = st.number_input("Times 60-89 Days Past Due", min_value=0, max_value=20, value=0, step=1)
    times_90_late = st.number_input("Times 90+ Days Past Due", min_value=0, max_value=20, value=0, step=1)
    open_credit = st.number_input("Number of Open Credit Lines", min_value=0, max_value=50, value=5, step=1)

with col_debt:
    st.subheader("🏠 Debt & Utilization")
    revolving_util = st.number_input("Revolving Utilization Rate", min_value=0.0, max_value=2.0, value=0.25, step=0.05)
    st.caption("*(Percentage of available credit used. >0.70 is high)*")
    debt_ratio = st.number_input("Debt Ratio", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
    st.caption("*(Percentage of income going to debt payments)*")
    real_estate_loans = st.number_input("Number of Real Estate Loans", min_value=0, max_value=20, value=1, step=1)

st.markdown("<br>", unsafe_allow_html=True) # Spacer

# --- PREDICT BUTTON ---
if st.button("📈 Run Risk Assessment", type="primary", use_container_width=True):
    
    # 1. Calculate Engineered Feature
    debt_to_income = debt_ratio * monthly_income
    
    # 2. Map inputs to exact CSV column names
    client_data = {}
    for col in feature_names:
        col_lower = col.lower()
        if 'utilization' in col_lower: client_data[col] = revolving_util
        elif 'age' in col_lower: client_data[col] = age
        elif '30_59' in col_lower or '30-59' in col_lower: client_data[col] = past_due_30_59
        elif 'debt_ratio' in col_lower: client_data[col] = debt_ratio
        elif 'income' in col_lower: client_data[col] = monthly_income
        elif 'open_credit' in col_lower or 'lines_and_loans' in col_lower: client_data[col] = open_credit
        elif '90' in col_lower: client_data[col] = times_90_late
        elif 'real_estate' in col_lower: client_data[col] = real_estate_loans
        elif '60_89' in col_lower or '60-89' in col_lower: client_data[col] = past_due_60_89
        elif 'dependent' in col_lower: client_data[col] = num_dependents
        elif 'debt_to_income' in col_lower: client_data[col] = debt_to_income
        else: client_data[col] = 0 
            
    df_client = pd.DataFrame([client_data])
    client_scaled = scaler.transform(df_client)
    prob_default = model.predict_proba(client_scaled)[0][1] * 100
    
    # 3. Business Logic
    if prob_default < 30:
        risk = "🟢 Low Risk"
        decision = "✅ APPROVED"
        loan_amount = max(0, (monthly_income * 3) - debt_to_income)
    elif prob_default < 60:
        risk = "🟠 Medium Risk"
        decision = "❌ REJECTED"
        loan_amount = 0
    else:
        risk = "🔴 High Risk"
        decision = "❌ REJECTED"
        loan_amount = 0

    # --- DISPLAY RESULTS ---
    st.markdown("---")
    st.subheader("📊 Assessment Results")
    
    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        st.metric(label="Default Probability", value=f"{prob_default:.2f}%")
    with res_col2:
        st.metric(label="Risk Classification", value=risk)
    with res_col3:
        if loan_amount > 0:
            st.metric(label="Decision & Limit", value=decision)
        else:
            st.metric(label="Decision", value=decision)
            
    # Visual Progress Bar for Probability
    st.progress(float(prob_default) / 100)
    
    # Custom Messages based on decision
    if decision == "✅ APPROVED":
        st.success(f"**Application Approved.** Based on the low-risk profile, the maximum recommended credit limit is **${loan_amount:,.2f}**.")
    else:
        st.error(f"**Application Denied.** The applicant's profile falls into the {risk} category. Extending credit at this time poses a high risk of capital loss.")
        with st.expander("💡 Why was this rejected? (Risk Factors Identified)"):
            reasons = []
            if revolving_util > 0.70: reasons.append("- High credit utilization (>70%)")
            if past_due_30_59 >= 2 or past_due_60_89 >= 1 or times_90_late >= 1: reasons.append("- History of severe late payments detected")
            if debt_ratio > 0.50: reasons.append("- Debt ratio exceeds safe threshold (>50%)")
            if monthly_income < 3000: reasons.append("- Low monthly income relative to requested debt")
            
            if reasons:
                for reason in reasons:
                    st.write(reason)
            else:
                st.write("- Combination of multiple minor risk factors triggered the rejection threshold.")