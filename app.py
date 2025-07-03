"""
CarMatchPro – Flask‑based ML Car Recommendation App
Author: Syed Saad Ali
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import numpy as np
import os

# ------------------------------------------------------------------------
# Paths & configuration
# ------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"

app = Flask(__name__, template_folder="templates", static_folder="static")

# Secret key:  set `SECRET_KEY` in your environment or .env file
app.secret_key = os.getenv("SECRET_KEY", "dev")      # Fallback 'dev' for local use

# Optional debug flag:  FLASK_DEBUG=true / false
DEBUG_MODE = os.getenv("FLASK_DEBUG", "false").lower() == "true"

# ------------------------------------------------------------------------
# Load models & data (gracefully falls back to mock mode)
# ------------------------------------------------------------------------
try:
    df       = pd.read_csv(DATA_DIR / "clustered_pakwheels.csv")
    kmeans   = joblib.load(MODEL_DIR / "kmeans_model.pkl")
    scaler   = joblib.load(MODEL_DIR / "scaler.pkl")
    le_fuel, le_trans, le_assembly = joblib.load(MODEL_DIR / "label_encoders.pkl")
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"⚠️  Error loading models: {e}")
    print("📝 Using mock data for demonstration")
    df = kmeans = scaler = le_fuel = le_trans = le_assembly = None

# ------------------------------------------------------------------------
# Recommendation logic
# ------------------------------------------------------------------------
def recommend_cars(price, mileage, engine_capacity, vehicle_age,
                   fuel_type, transmission, assembly, top_n=5):
    try:
        # If any model component is missing → use mock
        if any(m is None for m in [df, kmeans, scaler, le_fuel, le_trans, le_assembly]):
            return get_mock_recommendations()

        # Encode categorical features
        fuel_encoded     = le_fuel.transform([fuel_type])[0]
        trans_encoded    = le_trans.transform([transmission])[0]
        assembly_encoded = le_assembly.transform([assembly])[0]

        # Scale input
        user_input   = [[price, mileage, engine_capacity, vehicle_age,
                         fuel_encoded, trans_encoded, assembly_encoded]]
        user_scaled  = scaler.transform(user_input)

        # Predict cluster
        user_cluster      = kmeans.predict(user_scaled)[0]
        cluster_distances = kmeans.transform(user_scaled)[0]

        # Within‑cluster similarity
        feature_cols   = ['price', 'mileage', 'engine_capacity', 'vehicle_age',
                          'fuel_type_encoded', 'transmission_encoded', 'assembly_encoded']
        cluster_cars   = df[df['cluster'] == user_cluster].copy()
        cluster_scaled = scaler.transform(cluster_cars[feature_cols])

        # Weighted cosine similarity (price given extra weight)
        weights                   = np.array([2.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        user_scaled_weighted      = user_scaled * weights
        cluster_scaled_weighted   = cluster_scaled * weights
        similarities              = cosine_similarity(user_scaled_weighted,
                                                      cluster_scaled_weighted)[0]
        cluster_cars['similarity'] = similarities

        # Price filter: ±10 % (upper bound = user budget)
        price_range = (price * 0.9, price)
        cluster_cars = cluster_cars[
            (cluster_cars['price'] >= price_range[0]) &
            (cluster_cars['price'] <= price_range[1])
        ]

        # If empty → try next closest clusters
        if cluster_cars.empty:
            for next_cluster in np.argsort(cluster_distances)[1:]:
                cluster_cars = df[df['cluster'] == next_cluster].copy()
                cluster_scaled          = scaler.transform(cluster_cars[feature_cols])
                cluster_scaled_weighted = cluster_scaled * weights
                similarities            = cosine_similarity(user_scaled_weighted,
                                                             cluster_scaled_weighted)[0]
                cluster_cars['similarity'] = similarities
                cluster_cars = cluster_cars[
                    (cluster_cars['price'] >= price_range[0]) &
                    (cluster_cars['price'] <= price_range[1])
                ]
                if not cluster_cars.empty:
                    break

        if cluster_cars.empty:
            return get_mock_recommendations()

        # Top‑N recommendations
        top_matches = cluster_cars.sort_values(by='similarity', ascending=False).head(top_n)
        return [{
            'rank'          : i + 1,
            'title'         : row['title'],
            'price'         : f"PKR {int(row['price']):,}",
            'fuel_type'     : row['fuel_type'],
            'transmission'  : row['transmission'],
            'engine_capacity': f"{int(row['engine_capacity'])} cc",
            'vehicle_age'   : f"{int(row['vehicle_age'])} years",
            'mileage'       : f"{int(row['mileage']):,} km",
            'similarity'    : f"{row['similarity'] * 100:.1f}%"
        } for i, (_, row) in enumerate(top_matches.iterrows())]

    except Exception as e:
        print(f"Error in recommendation: {e}")
        return get_mock_recommendations()

# ------------------------------------------------------------------------
# Mock recommendations (shown when models are absent)
# ------------------------------------------------------------------------
def get_mock_recommendations():
    return [
        {
            'rank': 1,
            'title': "Toyota Corolla GLi 2018",
            'price': "PKR 2,500,000",
            'fuel_type': "Petrol",
            'transmission': "Automatic",
            'engine_capacity': "1300 cc",
            'vehicle_age': "6 years",
            'mileage': "14 km/l",
            'similarity': "98%"
        },
        {
            'rank': 2,
            'title': "Honda Civic VTi Oriel 2017",
            'price': "PKR 2,800,000",
            'fuel_type': "Petrol",
            'transmission': "Automatic",
            'engine_capacity': "1800 cc",
            'vehicle_age': "7 years",
            'mileage': "12 km/l",
            'similarity': "95%"
        }
    ]

# ------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html',
                           fuel_types=['Petrol', 'Diesel', 'CNG', 'Hybrid'],
                           transmissions=['Manual', 'Automatic'],
                           assemblies=['Local', 'Imported'])


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/results')
def results():
    return render_template('results.html')


@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.form
        recommendations = recommend_cars(
            int(data['price']),
            int(data['mileage']),
            int(data['engine_capacity']),
            int(data['vehicle_age']),
            data['fuel_type'],
            data['transmission'],
            data['assembly'],
            top_n=5
        )

        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'search_params': {
                'Budget'      : f"PKR {int(data['price']):,}",
                'Mileage'     : f"{data['mileage']} km/l",
                'Engine'      : f"{data['engine_capacity']} cc",
                'Age'         : f"{data['vehicle_age']} years",
                'Fuel'        : data['fuel_type'],
                'Transmission': data['transmission'],
                'Assembly'    : data['assembly']
            }
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Invalid input.'}), 400

# ------------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------------
if __name__ == '__main__':
    print("🚀 CarMatchPro is running at http://127.0.0.1:5000")
    app.run(debug=DEBUG_MODE, host='0.0.0.0', port=5000)
