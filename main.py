from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# We only try to import these if the file exists later
try:
    import joblib
    import pandas as pd
except ImportError:
    pass

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "demand_forecaster.pkl"

class PredictionInput(BaseModel):
    consumption: float
    current_stock: int
    min_required: int

@app.post("/predict")
async def predict(data: PredictionInput):
    # Check if your friend's file is there
    if os.path.exists(MODEL_PATH):
        # REAL AI MODE
        model = joblib.load(MODEL_PATH)
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)
        forecast = float(prediction[0])
        mode = "REAL_AI"
    else:
        # MOCK MODE (Until your friend wakes up)
        forecast = data.consumption * 1.15
        mode = "MOCK_LOGIC"

    return {
        "forecast": round(forecast, 2),
        "mode": mode,
        "status": "Online"
    }

@app.get("/")
def health():
    return {"status": "Backend Alive", "model_loaded": os.path.exists(MODEL_PATH)}
