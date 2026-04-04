## Key Steps
1. **Exploratory Data Analysis (EDA)**
   - Checked feature distributions, skewness, and outliers.
   - Target column is binary; handled class imbalance with SMOTE.

2. **Feature Reduction**  
   - **VIF-based reduction**: iteratively removed features with VIF > 10, reducing 95 → 30 features.

3. **Data Splitting & Scaling**
   - Stratified train-test split to maintain class ratio.  
   - SMOTE applied to training data only.  
   - Scaling using `StandardScaler` for Logistic Regression and KNN.

4. **Model Training & Evaluation**
   - Logistic Regression, KNN, Decision Tree, Gradient Boosting.  
   - Metrics: Recall, Precision, F1-Score, Accuracy, Confusion Matrix (FN/TP/FP/TN).  
   - Threshold tuning and hyperparameter tuning (KNN) performed to optimize recall.

5. **Results**
   - **KNN Tuned (manual features)** achieved **highest recall (0.8636)**.  
   - False negatives minimized to 6 — ideal for risk-sensitive tasks.  
   - Trade-off: Precision is low (~0.15), acceptable since catching bankrupt companies is priority.

6. **Conclusion**
   - **KNN Tuned model** is the best fit for this task.  
   - **Recall is the primary evaluation metric** due to the cost of Type 2 errors.  
   - Feature reduction improves model stability without sacrificing recall.  

---
