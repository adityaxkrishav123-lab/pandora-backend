import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# Load keys from the .env file we just made
load_dotenv()

app = FastAPI()

# 🛡️ Enable Dashboard to talk to Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 Friday's Brain Setup
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
    engine_source = ""

    # --- STEP 1: RUN THE MODEL (.pkl) ---
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            # Create the exact dataframe your model expects
            # Ensure column names match what you used in training
            input_data = pd.DataFrame([[data.consumption, data.current_stock, data.min_required]], 
                                     columns=['consumption', 'current_stock', 'min_required'])
            prediction = model.predict(input_data)
            forecast = float(prediction[0])
            engine_source = "Neural Model (.pkl)"
        except Exception as e:
            print(f"Model Error: {e}")
            forecast = data.consumption * 1.2 # Safety fallback
            engine_source = "Safety Fallback"
    else:
        # If model is missing, use "Smart Heuristics"
        forecast = data.consumption * 1.15
        engine_source = "Heuristic Logic"

    # --- STEP 2: GET FRIDAY'S ADVICE (Llama-3) ---
    try:
        chat = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are Friday, a sharp industrial AI assistant. Give 1 tactical sentence based on these numbers."},
                {"role": "user", "content": f"Item: {data.item_name}. Stock: {data.current_stock}. Forecasted: {forecast}. Min: {data.min_required}."}
            ],
            model="llama3-8b-8192",
        )
        advice = chat.choices[0].message.content
    except:
        advice = "Proceed with caution. Check supplier lead times."

    return {
        "item": data.item_name,
        "forecast": round(forecast, 2),
        "friday_advice": advice,
        "engine": engine_source
    }

@app.get("/")
def home():
    return {"status": "Friday Online", "brain": "Llama-3-8B", "model_found": os.path.exists(MODEL_PATH)}
