import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Streamlit App Title
st.title("Customer Churn Analysis by Harshith Reddy")

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Display first few rows
    st.subheader("First 5 rows of the dataset")
    st.write(df.head())

    # Dataset Information
    st.subheader("Dataset Info")
    buffer = []
    df.info(buf=buffer)
    st.text("\n".join(buffer))

    # Summary Statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Check for Missing Values
    st.subheader("Missing Values Count")
    st.write(df.isnull().sum())

    # Target Variable Analysis
    if 'Churn' in df.columns:
        st.subheader("Churn Value Counts")
        st.write(df['Churn'].value_counts())
        fig, ax = plt.subplots()
        sns.countplot(x='Churn', data=df, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("The dataset does not contain the 'Churn' column.")
    
    # Data Cleaning
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Encoding Categorical Variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Split Dataset
    if 'Churn' in df.columns:
        X = df.drop(columns=['Churn'])
        y = df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Training & Evaluation
        st.subheader("Model Performance")
        
        # Logistic Regression
        st.write("Logistic Regression")
        log_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        log_model.fit(X_train, y_train)
        y_pred_log = log_model.predict(X_test)
        st.write("Accuracy Score:", accuracy_score(y_test, y_pred_log))
        st.text("Classification Report:\n" + classification_report(y_test, y_pred_log))
        
        # Random Forest
        st.write("Random Forest Classifier")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        st.write("Accuracy Score:", accuracy_score(y_test, y_pred_rf))
        st.text("Classification Report:\n" + classification_report(y_test, y_pred_rf))

        # Correlation Matrix
        st.subheader("Feature Correlation Matrix")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.error("The dataset must contain the 'Churn' column for model training.")
else:
    st.info("Please upload a CSV file to proceed.")
