# 🚗 CarMatch Pro - Smart Car Recommendation System

[![Built with Flask](https://img.shields.io/badge/Framework-Flask-blue)](https://flask.palletsprojects.com/)  
[![Python](https://img.shields.io/badge/Python-3.7%2B-yellow)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 📖 Project Overview

**CarMatch Pro** is a web-based car recommendation system built using the Flask framework in Python.  
The app helps users find the best matching used cars based on their preferences like budget, mileage, engine capacity, vehicle age, fuel type, transmission, and assembly origin.

The system leverages machine learning models to cluster and recommend vehicles that closely align with user inputs, providing a smart and personalized car search experience.

---

## 🛠️ Technologies and Libraries Used

- **Flask** - Lightweight Python web framework to build the server and render dynamic HTML pages.
- **Pandas** - Data manipulation and analysis library to handle the car dataset.
- **Joblib** - For loading pre-trained machine learning models and label encoders.
- **scikit-learn** (`sklearn`) - Used for clustering (KMeans), scaling (StandardScaler), and similarity calculations (cosine similarity).
- **NumPy** - For numerical operations and array handling.
- **HTML/CSS/JS** - For front-end templates and styling served through Flask's template engine (`Jinja2`).

---

## ⚙️ Application Structure

- `app.py` - Main Flask application file containing routes, model loading, and recommendation logic.
- `templates/` - HTML templates for rendering pages like home, recommendation form, results, dashboard, about, etc.
- `static/` - Static assets such as CSS, JavaScript, and images.
- `data/clustered_pakwheels.csv` - Preprocessed vehicle dataset with cluster labels.
- `models/` - Contains serialized ML models and label encoders:
  - `kmeans_model.pkl` - KMeans clustering model.
  - `scaler.pkl` - Scaler for feature normalization.
  - `label_encoders.pkl` - Label encoders for categorical features (fuel, transmission, assembly).

---

## 🚀 Features

- **User Input:** Users provide their preferences including budget, mileage, engine size, vehicle age, fuel type, transmission type, and assembly origin.
- **Data Preprocessing:** Inputs are encoded and scaled using pre-trained label encoders and scalers.
- **Clustering:** KMeans clusters cars into groups; user input is matched to closest cluster.
- **Similarity Matching:** Cosine similarity with feature weighting prioritizes price but considers multiple attributes.
- **Dynamic Filtering:** Results are filtered dynamically by ±10% of user’s budget for relevant suggestions.
- **Fallback Handling:** If no cars match criteria in the closest cluster, the system searches adjacent clusters or returns mock recommendations.
- **REST API Endpoint:** `/get_recommendations` accepts POST requests and returns JSON with top car matches and search parameters.
- **User-Friendly Web Interface:** Multiple routes for home, recommendation form, results display, dashboard, and about page.

---

## 📂 How It Works - Core Components

### Model and Data Loading

- On startup, the app loads:
  - The car dataset (`clustered_pakwheels.csv`) with clusters.
  - The pre-trained KMeans clustering model.
  - Scalers and label encoders to transform user inputs and data features.

### Recommendation Logic (`recommend_cars` function)

- User inputs are label encoded and scaled.
- The model predicts the cluster closest to user inputs.
- Calculates cosine similarity between user input and cars in the cluster with weighted features.
- Filters cars dynamically by price range.
- Returns top N (default 5) cars sorted by similarity.

### Routes

| Route            | Method | Description                        |
| ---------------- | ------ | -------------------------------- |
| `/`              | GET    | Homepage                         |
| `/recommendation`| GET    | Car recommendation input form    |
| `/get_recommendations` | POST | API endpoint for recommendations (returns JSON) |
| `/results`       | GET    | Display recommendations (HTML)   |
| `/dashboard`     | GET    | Dashboard with insights           |
| `/about`         | GET    | About the project                 |

---

## 🛠️ Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/syedsaadali11/CarMatchPro.git
   cd CarMatchPro
