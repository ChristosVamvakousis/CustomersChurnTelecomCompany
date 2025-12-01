import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


file_path = '/home/chris/Desktop/Projects/Customers_Churn_2/Original_DataSet.csv'

# ---  Data Loading and Initial Cleaning ---
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Please check the path.")
    exit()

# Drop the unique identifier column
df = df.drop('customerID', axis=1)

# ---  Encoding Text Features ---

# Label Encoding (Binary Text Columns)
# Map columns with only two values to 1 or 0.

# Gender (Female=1, Male=0)
df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})

# Target (Churn: Yes=1, No=0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Binary columns that map 'Yes' to 1, 'No' to 0 (or similar)
binary_map_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies'] 

for col in binary_map_cols:
    # Maps 'Yes' to 1, and all 'No' variants ('No', 'No internet service', 'No phone service') to 0
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0})
    
# SeniorCitizen is already 0 or 1, ensure type consistency
df['SeniorCitizen'] = df['SeniorCitizen'].astype(int) 

# ---One-Hot Encoding (Multi-Category Columns) ---


# Identify all remaining text columns (these are the multi-category ones)
categorical_cols = df.select_dtypes(include=['object']).columns

# Apply One-Hot Encoding to: Contract, PaymentMethod, InternetService, MultipleLines, etc.
# pd.get_dummies automatically creates a new 0/1 column for every unique value in the listed columns.
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False) 


# ---  Prepare Data for Random Forest ---

# Separate Features (X) from the Target (y)
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Final check: ensure all features are numerical
X = X.select_dtypes(include=np.number) 


# ---  Run Random Forest Feature Importance ---

# Initialize the classifier and fit the model
model = RandomForestClassifier(n_estimators=500, random_state=42) # Tried at first with 100 and gave the same results
model.fit(X, y)

# Extract and sort feature importance scores
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
sorted_importances = feature_importances.sort_values(ascending=False)

# --- Visualization of Results ---

print("--- Top 15 Features Ranked by Random Forest Importance ---")
print(sorted_importances.head(15))

plt.figure(figsize=(10, 8))
sns.barplot(x=sorted_importances.head(15).values, y=sorted_importances.head(15).index, palette='viridis')
plt.title('Random Forest Feature Importance for Customer Churn')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--')
plt.show()
