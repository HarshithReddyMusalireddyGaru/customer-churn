import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# App Configuration
st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📊",
    layout="wide"
)

# Sidebar Content
st.sidebar.title("Welcome!")
st.sidebar.info(
    """
    **Developed by:** Harshith Reddy  
    **Email:** [mharshithreddy0@gmail.com](mailto:mharshithreddy0@gmail.com)
    """
)
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

# Main Content
st.title("Customer Churn Analysis")
st.header("Exploring and Predicting Churn for Telco Customers")

# Step 1: Upload and Load Dataset
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("First 5 Rows of the Dataset")
    st.write(df.head())

    st.subheader("Dataset Information")
    buffer = df.info(buf=None)
    st.text(buffer)

    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Step 2: Analyze Missing Values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Step 3: Analyze the Target Variable (Churn)
    st.subheader("Target Variable: Churn")
    if "Churn" in df.columns:
        st.write(df["Churn"].value_counts())
        fig, ax = plt.subplots()
        sns.countplot(x="Churn", data=df, ax=ax)
        st.pyplot(fig)
    else:
        st.error("The dataset does not contain a 'Churn' column!")

    # Step 4: Explore Numerical Features
    st.subheader("Numerical Feature Analysis")
    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numerical_columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    # Step 5: Explore Categorical Features
    st.subheader("Categorical Feature Analysis")
    categorical_columns = df.select_dtypes(include=["object"]).columns
    for col in ["gender", "InternetService", "Contract"]:
        if col in df.columns:
            fig, ax = plt.subplots()
            sns.countplot(x=col, data=df, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

    # Step 6: Data Cleaning and Preprocessing
    st.subheader("Data Cleaning and Preprocessing")
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        st.write("Missing values in TotalCharges after conversion:", df["TotalCharges"].isnull().sum())
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    st.write("Dataset after cleaning:")
    st.write(df.head())

    # Step 7: Model Training
    st.subheader("Model Training and Evaluation")
    irrelevant_cols = ["customerID", "PhoneService", "MultipleLines"]
    df = df.drop(columns=[col for col in irrelevant_cols if col in df.columns], errors="ignore")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression
    st.write("**Logistic Regression**")
    log_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)
    st.write("Accuracy Score:", accuracy_score(y_test, y_pred_log))
    st.text("Classification Report:\n" + classification_report(y_test, y_pred_log))

    # Random Forest
    st.write("**Random Forest Classifier**")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    st.write("Accuracy Score:", accuracy_score(y_test, y_pred_rf))
    st.text("Classification Report:\n" + classification_report(y_test, y_pred_rf))

    # Feature Correlation
    st.subheader("Feature Correlation Matrix")
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    st.pyplot()
else:
    st.warning("Please upload a dataset to proceed!")


