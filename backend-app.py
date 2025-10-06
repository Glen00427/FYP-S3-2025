# backend/app.py (Flask example)
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow requests from your React app

# Load your trained model
model = joblib.load('congestion_model.pkl')
features = joblib.load('feature_columns.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Extract route segments from 'from' and 'to'
    # (You'd geocode these to get LinkIDs from your training data)
    from_location = data['from']
    to_location = data['to']
    depart_time = datetime.fromisoformat(data.get('departTime', datetime.now().isoformat()))
    
    # Get route LinkIDs (simplified - you'd use actual routing logic)
    route_links = get_route_link_ids(from_location, to_location)
    
    # Aggregate features for this route
    route_features = aggregate_route_features(route_links, depart_time)
    
    # Predict congestion probability
    X = pd.DataFrame([route_features])[features]
    congestion_prob = model.predict_proba(X)[0][1]  # Probability of congestion
    
    # Generate alternative routes and predict for each
    alt_routes = generate_alternative_routes(from_location, to_location)
    predictions = []
    
    for alt_route in alt_routes:
        alt_features = aggregate_route_features(alt_route['links'], depart_time)
        alt_X = pd.DataFrame([alt_features])[features]
        alt_prob = model.predict_proba(alt_X)[0][1]
        predictions.append({
            'route_id': alt_route['id'],
            'route_name': alt_route['name'],
            'congestion_prob': float(alt_prob),
            'duration_min': alt_route['duration'],
            'distance_km': alt_route['distance']
        })
    
    # Sort by congestion probability
    predictions.sort(key=lambda x: x['congestion_prob'])
    
    return jsonify({
        'best_route_id': predictions[0]['route_id'],
        'best_route_name': predictions[0]['route_name'],
        'best_congestion_prob': predictions[0]['congestion_prob'],
        'best_confidence': 0.85,  # Calculate actual confidence
        'best_duration_min': predictions[0]['duration_min'],
        'best_distance_km': predictions[0]['distance_km'],
        'worst_route_id': predictions[-1]['route_id'],
        'worst_route_name': predictions[-1]['route_name'],
        'worst_congestion_prob': predictions[-1]['congestion_prob'],
        'worst_confidence': 0.82,
        'worst_duration_min': predictions[-1]['duration_min'],
        'worst_distance_km': predictions[-1]['distance_km']
    })

def aggregate_route_features(link_ids, depart_time):
    """
    Aggregate features across multiple road segments (like your notebook).
    Returns a dict matching your model's feature names.
    """
    # Fetch current speed data for these LinkIDs
    # Calculate aggregates: avg speed, min speed, incident counts, etc.
    return {
        'SpeedKMH_Est': 45.0,  # Average across segments
        'MinimumSpeed': 30.0,
        'MaximumSpeed': 60.0,
        'dow': depart_time.weekday(),
        'hour': depart_time.hour,
        'incident_count': 0,
        'vms_count': 0,
        'cctv_count': 36000,
        'ett_mean': 1.75
    }

if __name__ == '__main__':
    app.run(port=5000, debug=True)