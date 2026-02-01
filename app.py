import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Page Configuration
st.set_page_config(page_title="AI Resume Screening Dashboard", layout="wide")

# 1. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("AI_Resume_Screening.csv")
    return df

df = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Overview", "Analysis Visuals", "Model & Evaluation", "Candidate Prediction"])

# --- PAGE 1: DATASET OVERVIEW ---
if page == "Dataset Overview":
    st.title("📄 Dataset Overview")
    st.write("Current dataset contains resume details for AI-based screening.")
    st.dataframe(df.head(10))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Info")
        st.write(f"Total Rows: {df.shape[0]}")
        st.write(f"Total Columns: {df.shape[1]}")
    with col2:
        st.subheader("Missing Values")
        st.write(df.isnull().sum())

# --- PAGE 2: ANALYSIS VISUALS (Univariate & Bivariate) ---
elif page == "Analysis Visuals":
    st.title("📊 Analysis Visuals")
    
    # Univariate Analysis
    st.subheader("Univariate Analysis")
    fig1 = px.histogram(df, x="AI Score (0-100)", title="Distribution of AI Scores", color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig1)
    
    # Bivariate Analysis
    st.subheader("Bivariate Analysis")
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig2 = px.scatter(df, x="Experience (Years)", y="Salary Expectation ($)", color="Recruiter Decision",
                         title="Experience vs Salary Expectation")
        st.plotly_chart(fig2)
    
    with col_b:
        fig3 = px.box(df, x="Education", y="AI Score (0-100)", color="Education", 
                     title="AI Score Distribution by Education")
        st.plotly_chart(fig3)

# --- PAGE 3: MODEL BUILDING & EVALUATION ---
elif page == "Model & Evaluation":
    st.title("🤖 Model Comparison & Evaluation")
    
    # Simple Preprocessing for Model
    le = LabelEncoder()
    model_df = df.copy()
    model_df['Recruiter Decision'] = le.fit_transform(model_df['Recruiter Decision'])
    model_df['Education'] = le.fit_transform(model_df['Education'])
    model_df['Job Role'] = le.fit_transform(model_df['Job Role'])
    
    X = model_df[['Experience (Years)', 'Projects Count', 'AI Score (0-100)', 'Education']]
    y = model_df['Recruiter Decision']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model: Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    st.metric("Model Accuracy (Random Forest)", f"{acc*100:.2f}%")
    
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.table(pd.DataFrame(report).transpose())

# --- PAGE 4: PREDICTION SECTION ---
elif page == "Candidate Prediction":
    st.title("🔍 Predict Candidate Status")
    st.write("Enter details to check if the candidate is likely to be Hired or Rejected.")
    
    # Input Form
    with st.form("prediction_form"):
        name = st.text_input("Candidate Name")
        exp = st.slider("Experience (Years)", 0, 20, 5)
        projects = st.number_input("Projects Count", 0, 50, 3)
        ai_score = st.slider("AI Score (0-100)", 0, 100, 75)
        edu = st.selectbox("Education Level", df['Education'].unique())
        
        submit = st.form_submit_button("Predict Result")
        
        if submit:
            # Simple Logic based on AI Score & Experience (matching data patterns)
            # Pattern: Higher AI scores and Experience lead to 'Hire' 
            if ai_score >= 65 and exp >= 1:
                st.success(f"Prediction for {name}: **HIRE** ✅")
                st.balloons()
            else:
                st.error(f"Prediction for {name}: **REJECT** ❌")