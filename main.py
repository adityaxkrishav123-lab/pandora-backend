from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import joblib
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# 🛡️ CORS Policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 GROQ CLIENT (Set your GROQ_API_KEY in environment variables)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL_PATH = "demand_forecaster.pkl"

class PredictionInput(BaseModel):
    item_name: str
    consumption: float
    current_stock: int
    min_required: int

@app.post("/predict")
async def predict(data: PredictionInput):
    forecast = 0.0
    mode = ""

    # 1. 🔍 STEP 1: CALCULATE THE RAW FORECAST
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        # Assuming model was trained on [consumption, current_stock, min_required]
        input_df = pd.DataFrame([[data.consumption, data.current_stock, data.min_required]])
        prediction = model.predict(input_df)
        forecast = float(prediction[0])
        mode = "REAL_AI (.pkl)"
    else:
        forecast = data.consumption * 1.15
        mode = "MOCK_LOGIC (Heuristic)"

    # 2. ⚡ STEP 2: CONSULT LLAMA-3 (GROQ) FOR INSIGHTS
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are Friday, an expert industrial AI assistant. Give a concise, 1-sentence tactical recommendation based on inventory data."
                },
                {
                    "role": "user",
                    "content": f"Item: {data.item_name}. Stock: {data.current_stock}. Forecast: {forecast}. Min Required: {data.min_required}."
                }
            ],
            model="llama3-8b-8192",
        )
        friday_insight = chat_completion.choices[0].message.content
    except Exception as e:
        friday_insight = "Neural Link to Llama-3 unstable. Proceed with manual verification."

    return {
        "item": data.item_name,
        "forecasted_demand": round(forecast, 2),
        "friday_recommendation": friday_insight,
        "engine_mode": mode,
        "status": "Operational"
    }

@app.get("/")
def health():
    return {
        "status": "Friday Online",
        "local_model": os.path.exists(MODEL_PATH),
        "llama3_enabled": os.environ.get("GROQ_API_KEY") is not None
    })}
