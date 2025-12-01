import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 


file_path = '/home/chris/Desktop/Projects/Customers_Churn_2/Original_DataSet.csv'


df = pd.read_csv(file_path)
df = df.drop('customerID', axis=1)



# 2.1 Label Encoding (Strictly Binary Columns ONLY)
df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
df['SeniorCitizen'] = df['SeniorCitizen'].astype(int) 
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})

# 2.2 One-Hot Encoding (ALL Multi-Category Columns)
# This includes InternetService, Contract, PaymentMethod, and all services (OnlineSecurity, etc.)

categorical_cols = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False) 


# ---  Prepare Data for Random Forest  ---

# Convert the entire feature set to numeric, coercing any remaining issues.
# This forces the OHE columns to be treated as numbers (float/int).
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# CRITICAL FIX: Ensure all columns in X are numeric
# We use pd.to_numeric() on the entire set of features (X)
for col in X.columns:
    if X[col].dtype != np.number:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)


# ---  Run Random Forest Feature Importance ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Extract and sort feature importance scores
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
sorted_importances = feature_importances.sort_values(ascending=False)

# --- 5. Display the Full and Isolated Ranking ---
print("--- FULL Feature Ranking (CORRECTED RUN) ---")
print(sorted_importances.head(30)) 

print("\n--- Isolated Ranking of Critical OHE Features (CORRECTED RUN) ---")
ohe_ranking = sorted_importances[
    sorted_importances.index.str.contains('Contract_|InternetService_|PaymentMethod_|TechSupport_|OnlineSecurity_')
].head(15)
print(ohe_ranking)
