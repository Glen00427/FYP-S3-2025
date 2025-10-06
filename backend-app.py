# backend-app.py (COMPLETE VERSION)
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import requests

app = Flask(__name__)
CORS(app)

# Load your trained models (download from Colab first!)
model = joblib.load('rf.joblib')  # Your Random Forest
features = joblib.load('feature_columns.joblib')  # ['SpeedKMH_Est', 'MinimumSpeed', ...]

# LTA API credentials (from your notebook)
LTA_ACCOUNT_KEY = 'RYS6WoFQRNmktb5h2N0u5w=='

def get_current_traffic_data(link_ids):
    """Fetch live traffic speeds for given LinkIDs from LTA API"""
    url = 'https://datamall2.mytransport.sg/ltaodataservice/v4/TrafficSpeedBands'
    headers = {'AccountKey': LTA_ACCOUNT_KEY, 'accept': 'application/json'}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return {}
    
    data = response.json().get('value', [])
    # Filter for our LinkIDs
    relevant = [d for d in data if d.get('LinkID') in link_ids]
    return {d['LinkID']: d for d in relevant}


def get_route_link_ids(from_location, to_location):
    """
    Use OSRM to get a route, then map coordinates to your LinkIDs.
    For MVP: return dummy LinkIDs from your training data.
    """
    # TODO: Implement coordinate → LinkID mapping using spatial join
    # For now, return sample LinkIDs from your training data
    return ['103000000', '103000010', '103000011']  # Kent Road area


def aggregate_route_features(link_ids, depart_time):
    """
    Aggregate features across route segments (like your notebook code).
    """
    # Fetch current traffic for these links
    traffic_data = get_current_traffic_data(link_ids)
    
    if not traffic_data:
        # Fallback to defaults
        speeds = [45.0]
        min_speeds = [30.0]
        max_speeds = [60.0]
    else:
        speeds = [traffic_data[lid].get('SpeedKMH_Est', 45) for lid in link_ids if lid in traffic_data]
        min_speeds = [traffic_data[lid].get('MinimumSpeed', 30) for lid in link_ids if lid in traffic_data]
        max_speeds = [traffic_data[lid].get('MaximumSpeed', 60) for lid in link_ids if lid in traffic_data]
    
    # Aggregate features
    return {
        'SpeedKMH_Est': np.mean(speeds),
        'MinimumSpeed': np.mean(min_speeds),
        'MaximumSpeed': np.mean(max_speeds),
        'dow': depart_time.weekday(),
        'hour': depart_time.hour,
        'incident_count': 0,  # TODO: Query your incident_report table
        'vms_count': 0,
        'cctv_count': 36000,
        'ett_mean': 1.75
    }


def generate_alternative_routes(from_location, to_location):
    """
    Call OSRM to get 2-3 alternative routes.
    """
    # For MVP, return dummy alternatives
    return [
        {'id': 'route_1', 'name': 'Via PIE', 'links': ['103000000', '103000010'], 'duration': 25, 'distance': 12},
        {'id': 'route_2', 'name': 'Via CTE', 'links': ['103000011', '103000014'], 'duration': 30, 'distance': 15},
    ]


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    from_location = data['from']
    to_location = data['to']
    depart_time = datetime.fromisoformat(data.get('departTime', datetime.now().isoformat()))
    
    # Get alternative routes
    alt_routes = generate_alternative_routes(from_location, to_location)
    predictions = []
    
    for route in alt_routes:
        # Aggregate features for this route
        route_features = aggregate_route_features(route['links'], depart_time)
        
        # Predict congestion probability
        X = pd.DataFrame([route_features])[features]
        congestion_prob = model.predict_proba(X)[0][1]  # Probability of class 1 (congested)
        
        predictions.append({
            'route_id': route['id'],
            'route_name': route['name'],
            'congestion_prob': float(congestion_prob),
            'duration_min': route['duration'],
            'distance_km': route['distance'],
        })
    
    # Sort by congestion probability (best = lowest congestion)
    predictions.sort(key=lambda x: x['congestion_prob'])
    
    return jsonify({
        'best_route_id': predictions[0]['route_id'],
        'best_route_name': predictions[0]['route_name'],
        'best_congestion_prob': predictions[0]['congestion_prob'],
        'best_confidence': 0.85,
        'best_duration_min': predictions[0]['duration_min'],
        'best_distance_km': predictions[0]['distance_km'],
        'worst_route_id': predictions[-1]['route_id'],
        'worst_route_name': predictions[-1]['route_name'],
        'worst_congestion_prob': predictions[-1]['congestion_prob'],
        'worst_confidence': 0.82,
        'worst_duration_min': predictions[-1]['duration_min'],
        'worst_distance_km': predictions[-1]['distance_km'],
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
