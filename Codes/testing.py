import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('Services.csv')
correlation_matrix = df.corr()
churn_correlations = correlation_matrix['Churn'].sort_values(ascending=False).drop('Churn')
plt.figure(figsize=(8, 6))
sns.barplot(x=churn_correlations.values, y=churn_correlations.index, palette='plasma')
plt.title('Pearson Correlation of Encoded Features with Churn')
plt.xlabel('Pearson Correlation Coefficient (r)')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--')
plt.show()

print("\n--- Feature Correlations with Churn ---")
print(churn_correlations)
