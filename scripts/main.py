import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

# Resolution for execution subfolder paths imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_features import ApplyVIF
import model_train

# Generate OS-independent paths tracking back up to the main root workspace
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'datasets', 'Dataset.csv')
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, 'models')
MODEL_FILE_PATH = os.path.join(MODEL_OUTPUT_DIR, 'bankruptcy_model.pkl')

print(f"Reading dataset location matrix from: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()


x = df.drop(columns=['Bankrupt?'])
y = df['Bankrupt?']

# Split data cleanly before applying VIF loops to guarantee zero structural data leakages
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Executing iterative Multicollinearity reduction analysis (VIF)...")
x_train_cleaned, x_test_cleaned = ApplyVIF(x_train, x_test).apply_vif()

ll = x_train_cleaned.columns.tolist()
with open(os.path.join(BASE_DIR, 'models', 'vif_columns.txt'), 'w') as f:
    f.write('\n'.join(ll))

print("Initializing GridSearch optimization pipeline layers...")
best_model, results = model_train.optimized_knn(x_train_cleaned, x_test_cleaned, y_train, y_test)

# Verify deployment output folder matrices exist cleanly
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Export full pipeline package
print(f"Serializing full model orchestration pipeline to: {MODEL_FILE_PATH}")
with open(MODEL_FILE_PATH, "wb") as f:
    pickle.dump(best_model, f)

print("Deployment pkl artifact generated successfully using KNN Tuned.")
