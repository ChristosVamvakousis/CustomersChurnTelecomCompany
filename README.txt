=============================================================================================================================
================================================== Telecom Customer Churn Analysis ==========================================
=============================================================================================================================


The objective of this project was to analyze a telecom company's customer data to identify the primary factors driving churn (customer attrition). The goal was to develop actionable, data-backed strategies to improve customer retention and reduce the 26.5% churn rate.




=============================================================================================================================
===================================================== Technical Stack =======================================================
=============================================================================================================================


Language: Python 
Libraries: Pandas (Data Handling & Correlation), NumPy, Scikit-learn (Random Forest Model for Feature Importance), Matplotlib/Seaborn (Visualization)




=============================================================================================================================
============================================= Analysis Methodology (Quick Summary) ==========================================
=============================================================================================================================


1) Data Preparation: Performed cleaning, missing value imputation, and used One-Hot Encoding (OHE) on key categorical features (e.g., Contract Type).

2) Initial Correlation Analysis: Calculated precise correlation coefficients to establish initial relationships. This identified Month-to-Month Contract (r=0.667) as the largest risk enabler and highlighted services like Tech Support/Online Security as retention anchors (negative correlation).

2) Feature Importance: Used a Random Forest model to rank features, confirming that MonthlyCharges and Tenure are the top predictors.

3) Deep-Dive Correlation: Calculated precise correlation coefficients to isolate key risk enablers:

	i) Month-to-Month Contract (High correlation with churn, r=0.667).

	ii) Fiber Optic service (High risk due to price-to-value mismatch).

	iii) Tech Support/Online Security (Act as retention anchors, showing negative correlation).

	IV) Profile Synthesis: Defined the high-risk "Perfect Storm" customer profile: Low Tenure + Fiber Optic User + Month-to-Month Contract.
