import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Title and Header
st.title("Customer Churn Analysis")
st.header("Exploring and Predicting Churn for Telco Customers")
st.subheader("Developed by Harshith Reddy")

# Upload Dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.subheader("First 5 Rows of the Dataset")
    st.write(df.head())

    # Dataset Information
    st.subheader("Dataset Information")
    buffer = []
    df.info(buf=buffer)
    info_str = "\n".join(buffer)
    st.text(info_str)
    
    # Summary Statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Check for Missing Values
    st.subheader("Missing Values Count")
    st.write(df.isnull().sum())

    # Target Variable Analysis
    if 'Churn' in df.columns:
        st.subheader("Target Variable: Churn")
        st.write(df['Churn'].value_counts())
        fig, ax = plt.subplots()
        sns.countplot(x='Churn', data=df, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("The dataset does not contain the 'Churn' column.")

    # Data Cleaning and Preprocessing
    st.subheader("Data Cleaning and Preprocessing")
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        st.write("Missing values in TotalCharges after conversion:", df['TotalCharges'].isnull().sum())
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    else:
        st.warning("The dataset does not contain the 'TotalCharges' column.")
    
    # Encode Categorical Variables
    st.subheader("Encoding Categorical Variables")
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    st.write("Categorical Columns:", categorical_cols)
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    st.subheader("Dataset after Encoding and Cleaning")
    st.write(df.head())

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Splitting the Dataset
    if 'Churn' in df.columns:
        X = df.drop(columns=['Churn'])
        y = df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write(f"Training Features Shape: {X_train.shape}")
        st.write(f"Testing Features Shape: {X_test.shape}")

        # Logistic Regression
        st.subheader("Logistic Regression Model")
        lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        st.write("Accuracy Score:", accuracy_score(y_test, y_pred_lr))
        st.text("Classification Report:\n" + classification_report(y_test, y_pred_lr))
        st.text("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred_lr)))

        # Random Forest Classifier
        st.subheader("Random Forest Model")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        st.write("Random Forest Accuracy Score:", accuracy_score(y_test, y_pred_rf))
        st.text("Classification Report:\n" + classification_report(y_test, y_pred_rf))
        st.text("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred_rf)))

    else:
        st.error("The dataset must contain the 'Churn' column for model training.")



