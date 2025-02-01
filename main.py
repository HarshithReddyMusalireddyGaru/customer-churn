import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to preprocess dataset
def preprocess_dataset(df, target_col='Churn', numeric_col='TotalCharges'):
    # Ensure numeric conversion
    if numeric_col in df.columns:
        df[numeric_col] = pd.to_numeric(df[numeric_col], errors='coerce')
        df[numeric_col].fillna(df[numeric_col].median(), inplace=True)
    else:
        print(f"Warning: {numeric_col} column is missing. Adding a default column.")
        df[numeric_col] = 0  # Add default column

    # Encode target variable
    if target_col in df.columns:
        df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})
    else:
        print(f"Error: {target_col} column is missing.")
        return None  # Exit if the target column is missing

    # One-hot encoding for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

# Load the dataset
try:
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
except FileNotFoundError:
    print("Dataset file not found. Please check the path.")
    exit()

# Step 1: Preprocess the dataset
df = preprocess_dataset(df)
if df is None:
    exit()  # Exit if preprocessing failed

# Step 2: Drop irrelevant columns
irrelevant_cols = ['customerID', 'PhoneService', 'MultipleLines']
df = df.drop(columns=[col for col in irrelevant_cols if col in df.columns], errors='ignore')

# Step 3: Split the dataset into training and testing sets
X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("Logistic Regression:")
print("Accuracy Score:", accuracy_score(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest:")
print("Accuracy Score:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Visualize Correlations
correlation_matrix = X.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

