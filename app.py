from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'your-secret-key-change-this'

# Load models and data with error handling
try:
    df = pd.read_csv(r'D:\pakwheels\data\clustered_pakwheels.csv')
    kmeans = joblib.load(r'D:\pakwheels\models\kmeans_model.pkl')
    scaler = joblib.load(r'D:\pakwheels\models\scaler.pkl')
    le_fuel, le_trans, le_assembly = joblib.load(r'D:\pakwheels\models\label_encoders.pkl')
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading models: {e}")
    print("📝 Using mock data for demonstration")
    df = None
    kmeans = None
    scaler = None
    le_fuel = le_trans = le_assembly = None

def recommend_cars(price, mileage, engine_capacity, vehicle_age, fuel_type, transmission, assembly, top_n=5):
    try:
        if any(model is None for model in [df, kmeans, scaler, le_fuel, le_trans, le_assembly]):
            print("Error: One or more models are None")
            return get_mock_recommendations()
        
        # Log label encoder categories
        print(f"le_fuel.classes_: {le_fuel.classes_}")
        print(f"le_trans.classes_: {le_trans.classes_}")
        print(f"le_assembly.classes_: {le_assembly.classes_}")
        
        # Log input values
        print(f"Input values: price={price}, mileage={mileage}, engine_capacity={engine_capacity}, "
              f"vehicle_age={vehicle_age}, fuel_type={fuel_type}, transmission={transmission}, assembly={assembly}")
        
        fuel_encoded = le_fuel.transform([fuel_type])[0]
        trans_encoded = le_trans.transform([transmission])[0]
        assembly_encoded = le_assembly.transform([assembly])[0]

        user_input = [[price, mileage, engine_capacity, vehicle_age, fuel_encoded, trans_encoded, assembly_encoded]]
        user_scaled = scaler.transform(user_input)
        print(f"Scaled input: {user_scaled}")

        # Predict cluster and log distances to all clusters
        user_cluster = kmeans.predict(user_scaled)[0]
        print(f"Predicted cluster: {user_cluster}")
        cluster_distances = kmeans.transform(user_scaled)[0]
        print(f"Distances to clusters: {dict(zip(range(len(cluster_distances)), cluster_distances))}")

        cluster_cars = df[df['cluster'] == user_cluster].copy()
        feature_cols = ['price', 'mileage', 'engine_capacity', 'vehicle_age',
                        'fuel_type_encoded', 'transmission_encoded', 'assembly_encoded']
        cluster_scaled = scaler.transform(cluster_cars[feature_cols])

        # Apply weights to prioritize price
        weights = [2.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5]  # Higher weight for price
        user_scaled_weighted = user_scaled * weights
        cluster_scaled_weighted = cluster_scaled * weights
        similarities = cosine_similarity(user_scaled_weighted, cluster_scaled_weighted)[0]

        cluster_cars['similarity'] = similarities
        # Dynamic price filter: ±10% of input price
        price_range = (price * 0.9, price)
        print(f"Filtering cars with price between {price_range[0]:,} and {price_range[1]:,}")
        cluster_cars = cluster_cars[(cluster_cars['price'] >= price_range[0]) & 
                                   (cluster_cars['price'] <= price_range[1])]
        print(f"Number of cars after price filter: {len(cluster_cars)}")
        
        if len(cluster_cars) == 0:
            print("No cars found in price range. Falling back to closest cluster.")
            # Try next closest cluster
            sorted_clusters = np.argsort(cluster_distances)
            for next_cluster in sorted_clusters[1:]:
                cluster_cars = df[df['cluster'] == next_cluster].copy()
                cluster_scaled = scaler.transform(cluster_cars[feature_cols])
                cluster_scaled_weighted = cluster_scaled * weights
                similarities = cosine_similarity(user_scaled_weighted, cluster_scaled_weighted)[0]
                cluster_cars['similarity'] = similarities
                cluster_cars = cluster_cars[(cluster_cars['price'] >= price_range[0]) & 
                                           (cluster_cars['price'] <= price_range[1])]
                print(f"Checking cluster {next_cluster}: {len(cluster_cars)} cars")
                if len(cluster_cars) > 0:
                    break
            if len(cluster_cars) == 0:
                print("No cars found in any cluster. Using mock recommendations.")
                return get_mock_recommendations()

        top_matches = cluster_cars.sort_values(by='similarity', ascending=False).head(top_n)

        recommendations = [{
            'rank': i + 1,
            'title': row['title'],
            'price': f"PKR {int(row['price']):,}",
            'fuel_type': row['fuel_type'],
            'transmission': row['transmission'],
            'engine_capacity': f"{int(row['engine_capacity'])} cc",
            'vehicle_age': f"{int(row['vehicle_age'])} years",
            'mileage': f"{int(row['mileage']):,} km",
            'similarity': f"{row['similarity'] * 100:.1f}%"
        } for i, (_, row) in enumerate(top_matches.iterrows())]
        
        print("Recommendations:", [f"{rec['title']} ({rec['price']}, similarity={rec['similarity']})" for rec in recommendations])
        return recommendations
    except Exception as e:
        print(f"Error in recommendation: {e}")
        return get_mock_recommendations()

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

# ---- ROUTES ---- #
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
                'Budget': f"PKR {int(data['price']):,}",
                'Mileage': f"{data['mileage']} km/l",
                'Engine': f"{data['engine_capacity']} cc",
                'Age': f"{data['vehicle_age']} years",
                'Fuel': data['fuel_type'],
                'Transmission': data['transmission'],
                'Assembly': data['assembly']
            }
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Invalid input.'}), 400

# ---- MAIN ---- #
if __name__ == '__main__':
    print("🚀 CarMatch Pro is running at http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
