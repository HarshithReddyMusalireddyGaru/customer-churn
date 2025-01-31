import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Title and Header
st.title("Customer Churn Analysis")
st.header("Exploring and Predicting Churn for Telco Customers")

# Upload Dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.subheader("First 5 Rows of the Dataset")
    st.write(df.head())

    # Basic Dataset Info
    st.subheader("Dataset Information")
    buffer = df.info(buf=None)
    st.text(buffer)
    st.write("Summary Statistics:")
    st.write(df.describe())

    # Missing Values
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])

    # Target Variable Analysis
    st.subheader("Target Variable: Churn")
    st.write(df['Churn'].value_counts())
    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=df, ax=ax)
    st.pyplot(fig)

    # Numerical Feature Analysis
    st.subheader("Numerical Features Analysis")
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    st.write("Numerical Columns:", list(numerical_columns))
    for col in numerical_columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    # Categorical Feature Analysis
    st.subheader("Categorical Features Analysis")
    categorical_columns = df.select_dtypes(include=['object']).columns
    st.write("Categorical Columns:", list(categorical_columns))
    for col in ['gender', 'InternetService', 'Contract']:
        if col in df.columns:
            fig, ax = plt.subplots()
            sns.countplot(x=col, data=df, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

    # Data Cleaning and Preprocessing
    st.subheader("Data Cleaning and Preprocessing")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    st.write("Missing values in TotalCharges after conversion:", df['TotalCharges'].isnull().sum())
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, columns=['gender', 'Partner', 'Dependents', 'InternetService',
                                     'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                     'TechSupport', 'StreamingTV', 'StreamingMovies',
                                     'Contract', 'PaymentMethod', 'PaperlessBilling'], drop_first=True)
    st.write("Dataset after Encoding:")
    st.write(df.head())

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Splitting the Data
    st.subheader("Train-Test Split")
    df = df.drop(columns=['customerID', 'PhoneService', 'MultipleLines'])
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.write(f"Training Features Shape: {X_train.shape}")
    st.write(f"Testing Features Shape: {X_test.shape}")

    # Logistic Regression Model
    st.subheader("Logistic Regression Model")
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Random Forest Model
    st.subheader("Random Forest Model")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    st.write("Random Forest Accuracy Score:", accuracy_score(y_test, y_pred_rf))
    st.text("Classification Report for Random Forest:")
    st.text(classification_report(y_test, y_pred_rf))
