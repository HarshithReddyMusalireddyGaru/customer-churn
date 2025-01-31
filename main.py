import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Step 1: Load and View Basic Information
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Step 2: Check for Missing Values
print("\nMissing Values Count:")
print(df.isnull().sum())

# Step 3: Analyze the Target Variable (Churn)
print("\nChurn Value Counts:")
print(df['Churn'].value_counts())

sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

# Step 4: Explore Numerical Features
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
print("\nNumerical Columns:", numerical_columns)

for col in numerical_columns:
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Step 5: Explore Categorical Features
categorical_columns = df.select_dtypes(include=['object']).columns
print("\nCategorical Columns:", categorical_columns)

# Check unique values for each categorical column
for col in categorical_columns:
    print(f"{col}: {df[col].nunique()} unique values")

# Visualize some categorical features
for col in ['gender', 'InternetService', 'Contract']:
    if col in df.columns:  # Ensure the column exists
        sns.countplot(x=col, data=df)
        plt.title(f"Distribution of {col}")
        plt.show()

# Step 6: Check Correlations
# Filter the dataframe to include only numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix
correlation_matrix = numeric_df.corr()

# Visualize the correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Step 1: Convert TotalCharges to Numeric
# Replace spaces with NaN and convert to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check for NaN values after conversion
print("\nMissing values in TotalCharges after conversion:")
print(df['TotalCharges'].isnull().sum())

# Fill missing TotalCharges with the median (or another strategy)
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Step 2: Encode Categorical Variables
# Convert 'Churn' to binary (Yes=1, No=0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Use one-hot encoding for categorical columns
df = pd.get_dummies(df, columns=['gender', 'Partner', 'Dependents', 'InternetService',
                                  'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                  'TechSupport', 'StreamingTV', 'StreamingMovies',
                                  'Contract', 'PaymentMethod', 'PaperlessBilling'],
                    drop_first=True)

# Step 3: Verify the Dataset
print("\nDataset after encoding and cleaning:")
print(df.head())
print("\nDataset info after cleaning:")
print(df.info())


# Filter the dataframe to include only numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix
correlation_matrix = numeric_df.corr()

# Visualize the correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Data Cleaning and Encoding (your existing code)

# Add this at the end of your script
from sklearn.model_selection import train_test_split

# Drop irrelevant columns
df = df.drop(columns=['customerID', 'PhoneService', 'MultipleLines'])

# Separate features and target variable
X = df.drop(columns=['Churn'])
y = df['Churn']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the splits
print(f"Training Features Shape: {X_train.shape}")
print(f"Testing Features Shape: {X_test.shape}")
print(f"Training Target Shape: {y_train.shape}")
print(f"Testing Target Shape: {y_test.shape}")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the model

model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
print("Random Forest Accuracy Score:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report for Random Forest:\n", classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix for Random Forest:\n", confusion_matrix(y_test, y_pred_rf))
