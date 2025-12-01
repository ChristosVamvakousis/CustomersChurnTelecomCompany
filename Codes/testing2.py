import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('Services.csv')

# Ensure 'Churn' is a numerical 0/1 variable for correlation analysis
if df['Churn'].dtype == 'object':
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# ---  One-Hot Encoding the 'InternetService' Column ---

# Create binary columns (0 or 1) for each internet service type.
# We keep all dummies (drop_first=False) for clearer interpretation.
internet_service_dummies = pd.get_dummies(df['InternetService'], prefix='InternetService', drop_first=False)

# Concatenate the new dummy columns with the original DataFrame 
# and drop the original 'InternetService' column, as well as the column that had the 0, 1, 2 encoding 
df_encoded = pd.concat([df.drop('InternetService', axis=1), internet_service_dummies], axis=1)


# ---  Correlation Analysis ---

#  Calculate the correlation matrix 
correlation_matrix_encoded = df_encoded.corr()

# Extract the correlation values for 'Churn' from the new matrix
# This gives the isolated correlation for the one-hot encoded features.
churn_correlations_isolated = correlation_matrix_encoded['Churn'].sort_values(ascending=False).drop('Churn')

# ---  Focus and Visualization ---

# Identify the new Internet Service columns
internet_service_cols = [col for col in churn_correlations_isolated.index if col.startswith('InternetService_')]

print("\n--- Isolated Correlation for Internet Services vs. Churn ---")
print(churn_correlations_isolated.loc[internet_service_cols])

# Create a figure to visualize only the isolated internet service correlations
plt.figure(figsize=(8, 5))
sns.barplot(
    x=churn_correlations_isolated.loc[internet_service_cols].values, 
    y=churn_correlations_isolated.loc[internet_service_cols].index, 
    palette=['red' if val > 0 else 'green' for val in churn_correlations_isolated.loc[internet_service_cols].values]
)
plt.title('Isolated Correlation of Internet Service Types with Churn')
plt.xlabel('Pearson Correlation Coefficient (r)')
plt.ylabel('Internet Service Type')
plt.grid(axis='x', linestyle='--')
plt.show()
