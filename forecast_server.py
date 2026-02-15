import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS  # Added to allow React connection

app = Flask(__name__)
CORS(app) # This allows your frontend to access this API

# Load the brain
model = joblib.load('demand_forecaster (2).pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json 
        
        # The model expects these exact 7 features in this order
        features = ['consumption', 'current_stock', 'min_required', 'day', 'month', 'week', 'cost']
        
        # Create a dataframe for the model
        df = pd.DataFrame([data], columns=features)
        
        # Get prediction
        prediction = model.predict(df)[0]
        
        return jsonify({
            "status": "success",
            "forecasted_demand": round(float(prediction), 2),
            "recommendation": "Restock Required" if prediction > data['current_stock'] else "Stable"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    print("🚀 Pandora AI Brain Online on Port 5000")
    app.run(port=5000, debug=True)
