# %%
import pandas as pd

# %%
df  = pd.read_csv('/content/drive/MyDrive/ksr_python/ML/Dataset.csv')

# %%
df.head()

# %%
list(df.columns)

# %%
df['Bankrupt?'].value_counts(normalize=True) * 100

# %% [markdown]
# Imbalence Ratio

# %%
non_bank = df['Bankrupt?'].value_counts()[0]
bank = df['Bankrupt?'].value_counts()[1]
ratio = non_bank/bank
ratio

# %% [markdown]
# ibalence ratio is severe

# %%
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='Bankrupt?',data=df)
plt.title('Class imbalance')
plt.show()

# %% [markdown]
# Feature distribution

# %% [markdown]
# drop bankurpt coloumn as it is required only for class imbalance

# %%
x = df.drop(columns=['Bankrupt?'])
summ_df = x.describe().T.reset_index(names = 'cols')

# %%
summ_df

# %% [markdown]
# Checking the range, 75 percentile and skew to flag outliers

# %%
skew_df = x.skew().reset_index().rename(columns={'index': 'cols', 0: 'skew'})

# %%
skew_df

# %%
summ_df = summ_df.merge(skew_df,how ='left',on='cols')

# %%
summ_df['range'] = summ_df['max'] - summ_df['min']
summ_df['max_to_Q3'] = summ_df['max']/summ_df['75%']

# %%
summ_df

# %% [markdown]
# Getting highly skewed values

# %%
len(summ_df[abs(summ_df['skew'])>2])

# %%
len(summ_df[summ_df['max_to_Q3']>10])

# %% [markdown]
# Now start Bivariate Analysis

# %%
class_means = df.groupby('Bankrupt?').mean().T
class_means = class_means.reset_index(names=['Bankrupt?'])

# %%
class_means.columns

# %% [markdown]
# Calculate absolute mean difference

# %%
class_means['difference'] = abs(class_means[1] - class_means[0])
class_means.sort_values(by='difference', ascending=False).head(15)

# %% [markdown]
# The scale of values are too large so correlation between dependant and independant variable will be good

# %%
corr_df = df.corr()['Bankrupt?'].abs().sort_values(ascending=False)

# %%
corr_df.head(15)

# %%
feat =  corr_df.abs().sort_values(ascending = False)[1:6].index
feat

# %%
import seaborn as sns
import matplotlib.pyplot as plt
for col in feat:
  sns.boxenplot(data = df, x = 'Bankrupt?', y = col)
  plt.title(f'Bankrupt v/s {col}')
  plt.show()

# %% [markdown]
# #Detecting multicollinearity
# 
# this is helpful regression models, using the df 'X' which has dependant variable excluded

# %%
X = df.drop('Bankrupt?', axis=1)
y = df['Bankrupt?']

# %%
corr_matrix = X.corr().abs()
corr_matrix

# %% [markdown]
# consider only upper triangle fo the matrix to avoid redundancy

# %%
import numpy as np
upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# %%
upper

# %%
high_corr_feature = [col for col in upper.columns if any(upper[col] >0.85)]
high_corr_feature

# %%
corr_df_sorted = corr_df.abs().sort_values(ascending=False)
strong_features = corr_df_sorted[corr_df_sorted > 0.02].index

# %% [markdown]
# actually some features have good correlation with target variable but as they belong to similar class the correlation between variables of similar class is good which is not acceptable for logistic regression so it's better to have one variable from that particular class

# %%
selected = []
threshold = 0.85

for feature in strong_features:
    if feature == 'Bankrupt?':
        continue

    if all(abs(X[feature].corr(X[sel])) < threshold for sel in selected):
        selected.append(feature)

# %%
x_cleaned = x[selected]

# %%
x_cleaned

# %% [markdown]
# After multicollinearity analysis the less important features are removed and now the number of features are reduced to 45 from 96

# %% [markdown]
# #Data Splitting and Class Balancing

# %% [markdown]
# function which has all part of pipline for spplitting the model, traing and evaluating the metrics

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix,precision_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd
def model_pipeline(
    X,y,model,
    model_name = 'model',
    scale = False,
    use_smote = False,
    test_size = 0.2,
    random_state = 42):
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,
                                                   stratify=y,random_state=random_state)
  steps = []
  if scale:
    steps.append(('scale',StandardScaler()))
  if use_smote:
    steps.append(('smote',SMOTE(random_state=random_state)))
  steps.append(('model',model))

  pipeline = Pipeline(steps)
  pipeline.fit(X_train,y_train)
  y_pred = pipeline.predict(X_test)
  cm = confusion_matrix(y_test, y_pred)
  tn, fp, fn, tp = cm.ravel()
  results = {
        "Model": model_name,
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred),
        "False Negatives": fn,
        "True Positives": tp,
        "False Positives": fp,
        "True Negatives": tn
    }

  return pipeline, pd.DataFrame([results])


# %% [markdown]
# ##checking for df in which collinearity is removed manually

# %%
X_manual = x_cleaned.copy()
y = df['Bankrupt?']

# %% [markdown]
# ###Logistic regression

# %%
from sklearn.linear_model import LogisticRegression
logi = LogisticRegression(max_iter = 1000,class_weight="balanced")
model_log, result_log = model_pipeline( X_manual,y,
                                       model = logi,
                                       model_name = "Logistic Regression",
                                       scale = True,use_smote = True)
result_log

# %% [markdown]
# Good recall score, but as imbalence is being corrected twice once by using balanced in model and again applying smote so now results should be compared by different combinations

# %%
logi_None = LogisticRegression(max_iter = 1000,class_weight=None) # trying with class balance = None
model_log, result_log_w_o_balance = model_pipeline( X_manual,y,
                                       model = logi_None,
                                       model_name = "Logistic Regression not balanced",
                                       scale = True,use_smote = True)
result_log_w_o_balance


# %% [markdown]
# The recall is sameThis means smote has a significant effect in reducing class imbalence

# %% [markdown]
# ###KNN Now

# %%
from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier
#trying by consider 5 neighbours for now
knn_manual, res_knn_manual = model_pipeline(
    X_manual, y,
    model=KNeighborsClassifier(n_neighbors=5),
    model_name="KNN (Manual)",
    scale=True,
    use_smote=True
)

res_knn_manual

# %% [markdown]
# recall is less compared to logistic regression now tune the model using n_neighbous

# %%
from sklearn.model_selection import train_test_split

X_tr,X_tes,y_tr,y_tes = train_test_split(X_manual,y,test_size = 0.2,stratify = y,random_state=42)


# %%
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

knn_pipeline = Pipeline([
    ('scale',StandardScaler()),
    ('smote',SMOTE(random_state=42)),
    ('model',KNeighborsClassifier())

])


# %%
param_grid = {
    'model__n_neighbors': range(3, 21, 2)
}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(
    knn_pipeline,
    param_grid,
    scoring='recall',
    cv=5,
    n_jobs=-1
)

grid.fit(X_tr, y_tr)

print("Best k:", grid.best_params_)
print("Best CV Recall:", grid.best_score_)

# %%
best_knn = grid.best_estimator_

y_pred = best_knn.predict(X_tes)

from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score

cm = confusion_matrix(y_tes, y_pred)
tn, fp, fn, tp = cm.ravel()

print("Test Recall:", recall_score(y_tes, y_pred))
print("Test Precision:", precision_score(y_tes, y_pred))
print("Test F1:", f1_score(y_tes, y_pred))
print("Test Accuracy:", accuracy_score(y_tes, y_pred))
print("FN:", fn)
print("Confusion Matrix:\n", cm)
cm = confusion_matrix(y_tes, y_pred)
tn, fp, fn, tp = cm.ravel()

res_knn_tuned = pd.DataFrame([{
    "Model": "KNN Tuned (Manual)",
    "Recall": recall_score(y_tes, y_pred),
    "Precision": precision_score(y_tes, y_pred),
    "F1": f1_score(y_tes, y_pred),
    "Accuracy": accuracy_score(y_tes, y_pred),
    "False Negatives": fn,
    "True Positives": tp,
    "False Positives": fp,
    "True Negatives": tn
}])

res_knn_tuned

# %% [markdown]
# Recall is more than the previous model but precision is less

# %% [markdown]
# ###Decision tree

# %%
from sklearn.tree import DecisionTreeClassifier

dt_manual, res_dt_manual = model_pipeline(
    X_manual, y,
    model=DecisionTreeClassifier(random_state=42),
    model_name="Decision Tree (Manual)",
    scale=False,
    use_smote=True
)

res_dt_manual

# %% [markdown]
# recall is very low compared to KNN, it is lesser even than the baseline one so it is probably not worth it to fine tune

# %%
from sklearn.ensemble import RandomForestClassifier

rf_manual, res_rf_manual = model_pipeline(
    X_manual, y,
    model=RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),
    model_name="Random Forest (Manual)",
    scale=False,
    use_smote=True
)

res_rf_manual

# %% [markdown]
# try fine tuning the model

# %%
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from imblearn.pipeline import Pipeline

# x_tr,x_tes,y_tr,y_tes = train_test_split(X_manual,y,test_size=0.2,stratify=y,random_state = 42)
# rf_pipeline = Pipeline([
#     ('smote',SMOTE(random_state=42)),
#     ('model',RandomForestClassifier(random_state =42))
# ])
# rf_params_grid = {
#     'model__n_estimators' : [200,300],
#     'model__max_depth' : [None,10,20],
#     'model__min_samples_split' : [2,5] ,
#     'model__min_samples_leaf':[1,3]
# }

# %%
# grid_rf = GridSearchCV(estimator = rf_pipeline,
#                        param_grid = rf_params_grid,
#                        scoring = 'recall',cv=5, n_jobs =-1)
# grid_rf.fit(x_tr,y_tr)
# print("Best Params:", grid_rf.best_params_)
# print("Best CV Recall:", grid_rf.best_score_)

# %%
# best_rf = grid_rf.best_estimator_
# y_pred_rf = best_rf.predict(X_tes)

# from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score

# cm = confusion_matrix(y_tes, y_pred_rf)
# tn, fp, fn, tp = cm.ravel()

# print("Test Recall:", recall_score(y_tes, y_pred_rf))
# print("Test Precision:", precision_score(y_tes, y_pred_rf))
# print("FN:", fn)
# print("Confusion Matrix:\n", cm)

# %% [markdown]
# ###Gradient Boosting

# %%
from sklearn.ensemble import GradientBoostingClassifier

gb_manual, res_gb_manual = model_pipeline(
    X_manual, y,
    model=GradientBoostingClassifier(random_state=42),
    model_name="Gradient Boosting (Manual)",
    scale=False,
    use_smote=True
)

res_gb_manual

# %% [markdown]
# ###Ensembling

# %%
from sklearn.ensemble import VotingClassifier
voting_model = VotingClassifier(
    [
        ('knn',best_knn),
        ('log',LogisticRegression(max_iter=1000)),
        ('GB',GradientBoostingClassifier(random_state=42))
    ]

)
voting_model.fit(X_tr,y_tr)
results_ensemble = {
    "Model": "Voting Ensemble",
    "Recall": recall_score(y_tes, y_pred),
    "Precision": precision_score(y_tes, y_pred),
    "F1": f1_score(y_tes, y_pred),
    "Accuracy": accuracy_score(y_tes, y_pred),
    "False Negatives": fn,
    "True Positives": tp,
    "False Positives": fp,
    "True Negatives": tn
}

results_ensemble = pd.DataFrame([results_ensemble])
results_ensemble

# %% [markdown]
# ###Leaderboards

# %%
manual_results = pd.concat([
    result_log,
    res_knn_manual,
    res_knn_tuned,
    res_dt_manual,
    res_gb_manual,
    results_ensemble
], ignore_index=True)

manual_results.sort_values(by="Recall", ascending=False)

# %% [markdown]
# Among manually selected feature models, Tuned KNN achieved the highest recall (0.86) with the lowest false negatives (6). Since missing a bankrupt company is more costly than flagging a healthy one, KNN Tuned was selected as the best-performing model under the defined business objective(Minimize Type-2 errors).

# %%
len(x.columns)

# %%
x_vif = x.copy()


# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# Copy feature matrix
x_vif = x.copy()

# Function to compute VIF
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data['Features'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# Iteratively remove features with VIF > 10
while True:
    vif_data = calculate_vif(x_vif)
    max_vif = vif_data['VIF'].max()
    if max_vif > 10:
        drop_feat = vif_data.sort_values('VIF', ascending=False).iloc[0]['Features']
        print(f"Dropping {drop_feat} with VIF={max_vif:.2f}")
        x_vif = x_vif.drop(columns=[drop_feat])
    else:
        break

print(f"Remaining features: {x_vif.shape[1]}")

# %%
len(x_vif.columns)

# %% [markdown]
# Features reduced significantly compared to the manual fearure reduction

# %% [markdown]
# Check using KNN and logistic regression as they erformed better for manual one

# %%
from sklearn.linear_model import LogisticRegression

log_vif = LogisticRegression(max_iter=1000)
mod_log , res_log_vif = model_pipeline(
    x_vif,y,model = log_vif,
    model_name = 'Logistic Regression (VIF)',
    scale = True,
    use_smote = True
)
len(x_vif.columns)

# %%
res_log_vif

# %% [markdown]
# After reducing the feature set from 95 to 30 using VIF, Logistic Regression maintained the same recall (~0.795), ensuring we still catch most bankrupt companies. Precision dropped slightly due to fewer features providing discriminative power, but the model is now more stable and interpretable

# %% [markdown]
# ###KNN (BASELINE)

# %%
from sklearn.neighbors import KNeighborsClassifier
knn_vif, result_knn_vif = model_pipeline(x_vif,y,model = KNeighborsClassifier(n_neighbors=20), #considerig 20 hoing more neighbours might increase recall
                                         model_name = 'KNN VIF Baseline',
                                         scale = True,
                                         use_smote = True)


# %%
result_knn_vif

# %% [markdown]
# After reducing features from 95 to 30 using VIF, KNN baseline maintained the same recall (0.8636) and FN (6), which is our primary metric. Precision dropped slightly because some redundant features that helped distinguish healthy companies were removed. This shows that VIF-based reduction keeps recall intact while stabilizing the model.

# %%
vif_results_df = pd.concat([res_log_vif,result_knn_vif])

# %%
vif_results_df.sort_values('Recall',ascending=False)

# %% [markdown]
# So, both mannually removed features and feature removal using VIF have yielded same results

# %%
res_df_comb = pd.concat([manual_results,vif_results_df]).sort_values('Recall',ascending=False)

# %%
res_df_comb

# %%
metrics = ["Recall", "Precision", "F1", "Accuracy"]
all_results_melted = res_df_comb.melt(
    id_vars="Model",
    value_vars=metrics,
    var_name="Metric",
    value_name="Value"
)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(16,6))
sns.barplot(x='Model',y='Value',hue='Metric', data=all_results_melted
)


# %% [markdown]
# Fom these we can conclude that the KNN Model is best fit for the Bankruptcy detection given that in this case we need to reduce the type 2 errors<br>
# Also here we need to consider recall as the main evaluation metric as reporting a false negative might result in heavy losses for the investors

# %%



