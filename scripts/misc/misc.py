import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix,precision_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import pickle


def model_pipeline(
    X_train,y_train,X_test ,y_test,model,
    model_name = 'model',
    scale = False,
    use_smote = False,
    random_state = 42):
  
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


def call_LR(x_train,x_test,y_train,y_test):
    log_vif = LogisticRegression(max_iter=1000)
    mod_log , res_log_vif = model_pipeline(
        X_train=x_train,
        X_test=x_test,
        y_train=y_train,
        y_test=y_test,
        model = log_vif,
        model_name = 'Logistic Regression (VIF)',
        scale = True,
        use_smote = True
    )
    return mod_log, res_log_vif

def call_KNN(x_train,x_test,y_train,y_test):

  knn_manual, res_knn_manual = model_pipeline(
      X_train=x_train,
      X_test=x_test,
      y_train=y_train,
      y_test=y_test,
      model = KNeighborsClassifier(),
      model_name = 'KNN (Manual)',
      scale = True,
      use_smote = True
      
  )
  return knn_manual, res_knn_manual

def model_DT(x_train,x_test,y_train,y_test):
    dt_vif = DecisionTreeClassifier(random_state=42)
    mod_dt, res_dt_vif = model_pipeline(
        X_train=x_train,
        X_test=x_test,
        y_train=y_train,
        y_test=y_test,
        model = dt_vif,
        model_name = 'Decision Tree (VIF)',
        scale = True,
        use_smote = True
    )
    return mod_dt, res_dt_vif

def model_RF(x_train,x_test,y_train,y_test):
    rf_vif = RandomForestClassifier(n_estimators=100, random_state=42)
    mod_rf, res_rf_vif = model_pipeline(
        X_train=x_train,
        X_test=x_test,
        y_train=y_train,
        y_test=y_test,
        model = rf_vif,
        model_name = 'Random Forest (VIF)',
        scale = True,
        use_smote = True
    )
    return mod_rf, res_rf_vif

def model_XGB(x_train,x_test,y_train,y_test):
    xgb_vif = GradientBoostingClassifier(random_state=42)
    mod_xgb, res_xgb_vif = model_pipeline(
        X_train=x_train,
        X_test=x_test,
        y_train=y_train,
        y_test=y_test,
        model = xgb_vif,
        model_name = 'XGBoost (VIF)',
        scale = True,
        use_smote = True
    )
    return mod_xgb, res_xgb_vif
