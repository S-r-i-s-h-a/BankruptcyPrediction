import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

def optimized_knn(x_train, x_test, y_train, y_test):
    knn_pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', KNeighborsClassifier())
    ])
    
    param_grid = {
        'model__n_neighbors': range(3, 21, 2)
    }
    
    grid = GridSearchCV(
        knn_pipeline,
        param_grid,
        scoring='recall',
        cv=5,
        n_jobs=-1
    )

    grid.fit(x_train, y_train)
    best_knn = grid.best_estimator_

    # Performance Inference Metrics Evaluation
    y_pred = best_knn.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "="*40 + "\nMODEL EVALUATION METRICS\n" + "="*40)
    print(f"Best Optimal K-Neighbors: {grid.best_params_['model__n_neighbors']}")
    print(f"Test Recall Metric:       {recall_score(y_test, y_pred):.4f}")
    print(f"Test Precision Metric:    {precision_score(y_test, y_pred):.4f}")
    print(f"Test F1 Evaluation:       {f1_score(y_test, y_pred):.4f}")
    print(f"Test Accuracy Score:      {accuracy_score(y_test, y_pred):.4f}")
    print(f"False Negatives (Risk):   {fn}")
    print(f"Confusion Matrix Array:\n{cm}\n" + "="*40)

    res_knn_tuned = pd.DataFrame([{
        "Model": "KNN Tuned",
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred),
        "False Negatives": fn,
        "True Positives": tp,
        "False Positives": fp,
        "True Negatives": tn
    }])
    
    return best_knn, res_knn_tuned
