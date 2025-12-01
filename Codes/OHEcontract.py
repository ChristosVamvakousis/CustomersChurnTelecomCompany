import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# 1000 data points for better correlation calculation
N = 1000
np.random.seed(42)

# Simulate Churn: High when Contract is Month-to-month and Low otherwise
churn = np.concatenate([
    np.ones(int(N*0.3)),  # 30% are Month-to-Month and Churn
    np.zeros(int(N*0.7))
])
np.random.shuffle(churn)

# Month-to-Month: Highly correlated with Churn
contract_m2m = np.where(churn == 1, 1, np.random.choice([0, 1], N, p=[0.75, 0.25])) 
contract_m2m = np.where(contract_m2m == 1, 1, 0) # Ensure it's 0 or 1

# One-year/Two-year: Negatively correlated with Churn
contract_1yr = np.where(churn == 0, 1, np.random.choice([0, 1], N, p=[0.85, 0.15]))
contract_2yr = np.where(churn == 0, 1, np.random.choice([0, 1], N, p=[0.90, 0.10]))


data = {
    'Churn': churn,
    'Contract_Month-to-month': contract_m2m,
    'Contract_One year': contract_1yr,
    'Contract_Two year': contract_2yr
}
df_encoded = pd.DataFrame(data).astype(int)
# -----------------------------------------------------------------------


# 1. Define the columns to analyze (Contract types and the Target)
contract_cols = [
    'Contract_Month-to-month', 
    'Contract_One year', 
    'Contract_Two year', 
    'Churn'
]

# 2. Subset the DataFrame
subset_df = df_encoded[contract_cols]

# 3. Calculate the correlation matrix
correlation_matrix = subset_df.corr()

# 4. Extract only the correlations with the 'Churn' column and sort them
# We sort to put the strongest relationship at the top
churn_correlations = correlation_matrix[['Churn']].drop('Churn').sort_values(by='Churn', ascending=False)

# 5. Create the focused heatmap visualization
plt.figure(figsize=(8, 6))
sns.heatmap(
    churn_correlations, 
    annot=True, 
    # Use 'coolwarm' for standard correlation display: 
    # RED for negative (good/loyal), BLUE for positive (bad/churn)
    cmap='coolwarm', 
    vmin=-0.5, # Set limits to better show contrast
    vmax=0.5,
    linewidths=0.5, 
    linecolor='black',
    cbar=True, 
    fmt=".3f", 
    annot_kws={"size": 14, "weight": "bold", "color": "black"} # Ensure text is visible
)
plt.title('Correlation of Contract Type vs. Churn', fontsize=16)
plt.ylabel('Contract Type (OHE)', fontsize=14)
plt.yticks(rotation=0)
plt.xticks([]) # Hide X-axis label "Churn" since it's obvious
plt.tight_layout()
plt.show()
