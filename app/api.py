from fastapi import FastAPI, HTTPException,UploadFile, File
import io
from pydantic import BaseModel
from typing import Dict
import pandas as pd
import json
import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS_PATH = os.path.join(BASE_DIR, 'inputs', 'input.json')
FEATURES_PATH = os.path.join(BASE_DIR, 'models', 'vif_columns.txt')
MODEL_FILE_PATH = os.path.join(BASE_DIR, 'models', 'bankruptcy_model.pkl')

app = FastAPI(title="Bankruptcy Prediction API")

class BankruptcyPrediction(BaseModel):
    metrics: Dict[str, float]

# Load required feature names and model once at startup
with open(FEATURES_PATH, 'r') as f:
    vif_columns = [line.strip() for line in f.read().splitlines() if line.strip()]

with open(MODEL_FILE_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

@app.get("/")
def health_check():
    return {"status": "online", "expected_features_count": len(vif_columns)}

@app.post("/predict-file")
async def predict_from_file(file: UploadFile = File(...)):
    contents = await file.read()

    if file.filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(contents))
    elif file.filename.endswith(".json"):
        raw = json.loads(contents.decode("utf-8"))
        # Supports either {"metrics": {...}} or a flat dict/list of dicts
        if isinstance(raw, dict) and "metrics" in raw:
            df = pd.DataFrame([raw["metrics"]])
        elif isinstance(raw, dict):
            df = pd.DataFrame([raw])
        else:
            df = pd.DataFrame(raw)
    else:
        raise HTTPException(status_code=400, detail="Only .csv or .json files are supported.")

    missing_features = [col for col in vif_columns if col not in df.columns]
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={"error": "Missing required columns", "missing_features": missing_features},
        )

    input_df = df[vif_columns]
    predictions = model.predict(input_df).astype(int).tolist()
    probabilities = model.predict_proba(input_df)[:, 1].round(4).tolist()

    return {
        "filename": file.filename,
        "rows_processed": len(df),
        "results": [
            {
                "row": i + 1,
                "bankrupt_prediction": pred,
                "risk_label": "High Risk (Bankrupt)" if pred == 1 else "Low Risk (Non-Bankrupt)",
                "bankruptcy_probability": prob,
            }
            for i, (pred, prob) in enumerate(zip(predictions, probabilities))
        ],
    }