import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = '/home/chris/Desktop/Projects/Customers_Churn_2/Original_DataSet.csv'

# ---  Data Loading and Preparation ---
print("--- Starting Data Loading and Visualization ---")
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Please check the path.")
    exit()

# Ensure 'Churn' column is stripped of any hidden whitespace
df.columns = df.columns.str.strip()

# ---  Convert Target Column to Numeric and then to String for Plotting ---
df['Churn_Status'] = df['Churn'].map({'Yes': '1', 'No': '0'}) # Use a new column as string keys

# =================================================================
# ============= Churn Count Bar Chart =============================
# =================================================================

churn_counts = df['Churn_Status'].value_counts().sort_index()

plt.figure(figsize=(7, 6))

# FIX: Palette keys are now strings '0' and '1' to match the index of churn_counts
palette = {'0': '#4daf4a', '1': '#e41a1c'} # Green for No Churn, Red for Churn

# To avoid the FutureWarning, we explicitly assign data, x, and y
ax = sns.barplot(x=churn_counts.index, y=churn_counts.values, palette=palette) 
plt.xticks([0, 1], ['No Churn', 'Churn'])
plt.title('Count of Customers Who Churned vs. Did Not Churn', fontsize=16)
plt.xlabel('Churn Status')
plt.ylabel('Customer Count')

# Add count labels on bars for easy reading
for i, count in enumerate(churn_counts.values):
    plt.text(i, count + 100, str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.show()


# =================================================================
# ============ Churn Percentage Pie Chart =========================
# =================================================================

# We use the original Churn column (Yes/No) for cleaner pie chart labels
churn_percentages = df['Churn'].value_counts(normalize=True) * 100

plt.figure(figsize=(7, 6))

labels = ['No Churn', 'Churn']
colors = ['#4daf4a', '#e41a1c'] 

plt.pie(churn_percentages, 
        labels=labels, 
        colors=colors, 
        autopct='%1.1f%%', 
        startangle=90, 
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5},
        textprops={'fontsize': 14}) 

plt.title('Percentage of Customer Churn', fontsize=16)
plt.show()

print("\n--- Churn Distribution Visualizations Complete ---")
