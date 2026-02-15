import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# Import logic from your existing server file
# Assuming your forecast_server.py has a function or model variable
try:
    from forecast_server import model as local_model
except ImportError:
    local_model = None

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    # 1. 🔍 Try the .pkl model (Local Intelligence)
    if os.path.exists(MODEL_PATH):
        try:
            # Load model if not imported
            m = local_model if local_model else joblib.load(MODEL_PATH)
            
            # Prepare data as per your specific model requirements
            # We use a DataFrame as it's the standard for .pkl models
            df = pd.DataFrame([[data.consumption, data.current_stock, data.min_required]], 
                              columns=['consumption', 'current_stock', 'min_required'])
            
            prediction = m.predict(df)
            forecast = float(prediction[0])
            mode = "REAL_AI (.pkl)"
        except Exception as e:
            forecast = data.consumption * 1.2
            mode = "FALLBACK_LOGIC"
    else:
        forecast = data.consumption * 1.15
        mode = "MOCK_LOGIC"

    # 2. ⚡ Get Friday's Tactical Insight (Llama-3)
    try:
        chat = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are Friday, the AI from Iron Man. You are managing a factory. Give a one-sentence tactical advice based on the inventory data provided. Be professional yet sharp."},
                {"role": "user", "content": f"Component: {data.item_name}. Stock: {data.current_stock}. Forecasted Demand: {forecast}. Min Safety: {data.min_required}."}
            ],
            model="llama3-8b-8192",
        )
        friday_advice = chat.choices[0].message.content
    except:
        friday_advice = "Neural Link unstable. Suggest maintaining current production pace."

    return {
        "item": data.item_name,
        "forecast": round(forecast, 2),
        "friday_advice": friday_advice,
        "engine_mode": mode
    }

@app.get("/")
def health():
    return {"status": "Friday Online", "llama3": "Linked", "local_model": os.path.exists(MODEL_PATH)}
