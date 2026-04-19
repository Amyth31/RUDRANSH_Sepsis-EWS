"""
SepsisAlert - Flask Backend
Routes: / (dashboard), /predict (prediction page), /map (hospital finder)
"""

from flask import Flask, request, jsonify, render_template
import pickle, json, os
import numpy as np
from urllib import parse, request as urlrequest
from urllib.error import HTTPError, URLError
import time
import math

app = Flask(__name__, template_folder='templates', static_folder='static')

BASE      = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, 'model')
with open(os.path.join(MODEL_DIR, 'metrics.json')) as f:
    metrics = json.load(f)

RISK_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}
RISK_COLORS = {0: '#22c55e', 1: '#f59e0b', 2: '#ef4444'}
OVERPASS_ENDPOINTS = [
    'https://overpass-api.de/api/interpreter',
    'https://overpass.private.coffee/api/interpreter',
    'https://maps.mail.ru/osm/tools/overpass/api/interpreter',
]
gb_model = None
rf_model = None
scaler = None
feature_cols = None
HOSPITAL_CACHE = {}
CACHE_TTL_SECONDS = 600

def ensure_prediction_assets():
    global gb_model, rf_model, scaler, feature_cols
    if gb_model is None:
        with open(os.path.join(MODEL_DIR, 'gb_model.pkl'), 'rb') as f:
            gb_model = pickle.load(f)
    if rf_model is None:
        with open(os.path.join(MODEL_DIR, 'rf_model.pkl'), 'rb') as f:
            rf_model = pickle.load(f)
    if scaler is None:
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
    if feature_cols is None:
        with open(os.path.join(MODEL_DIR, 'feature_cols.json')) as f:
            feature_cols = json.load(f)

def compute_features(vitals_6h):
    ensure_prediction_assets()
    keys = ['HR','MAP','SBP','Temp','O2Sat','Lactate','Creatinine','Platelets']
    feat = {}
    t = np.arange(6)
    for k in keys:
        s = np.array(vitals_6h[k], dtype=float)
        feat[f'mean_{k}']  = float(np.mean(s))
        feat[f'trend_{k}'] = float(np.polyfit(t, s, 1)[0])
    feat['shock_index'] = feat['mean_HR'] / max(feat['mean_SBP'], 1)
    return np.array([feat[c] for c in feature_cols])

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def nominatim_fallback(lat, lng):
    lat_delta = 10.0 / 111.0
    lon_scale = max(math.cos(math.radians(lat)), 0.2)
    lon_delta = 10.0 / (111.0 * lon_scale)
    min_lat = lat - lat_delta
    max_lat = lat + lat_delta
    min_lng = lng - lon_delta
    max_lng = lng + lon_delta
    viewbox = f'{min_lng},{max_lat},{max_lng},{min_lat}'
    base_url = 'https://nominatim.openstreetmap.org/search'
    collected = []
    seen = set()

    for query in ['hospital', 'clinic']:
        params = parse.urlencode({
            'format': 'jsonv2',
            'q': query,
            'limit': 50,
            'bounded': 1,
            'viewbox': viewbox,
        })
        req = urlrequest.Request(
            f'{base_url}?{params}',
            headers={
                'User-Agent': 'Rudransh-Sepsis-EWS/1.0',
                'Accept': 'application/json',
            },
            method='GET',
        )
        with urlrequest.urlopen(req, timeout=8) as response:
            results = json.loads(response.read().decode('utf-8'))
        for item in results:
            try:
                item_lat = float(item['lat'])
                item_lng = float(item['lon'])
            except (KeyError, ValueError, TypeError):
                continue
            if haversine_km(lat, lng, item_lat, item_lng) > 10.0:
                continue
            key = (round(item_lat, 5), round(item_lng, 5), item.get('display_name', ''))
            if key in seen:
                continue
            seen.add(key)
            display_name = item.get('display_name', '')
            name = display_name.split(',')[0].strip() if display_name else 'Hospital'
            collected.append({
                'lat': item_lat,
                'lon': item_lng,
                'tags': {
                    'name': name or 'Hospital',
                    'amenity': query,
                    'addr:street': display_name,
                }
            })
    return collected

@app.route('/')
def dashboard():
    return render_template('dashboard.html', metrics=metrics)

@app.route('/predict')
def predict_page():
    return render_template('index.html', metrics=metrics)

@app.route('/map')
def map_page():
    return render_template('map.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data    = request.get_json()
        vitals  = data.get('vitals')
        feat_vec   = compute_features(vitals)
        feat_scaled = scaler.transform(feat_vec.reshape(1, -1))
        prob_gb  = gb_model.predict_proba(feat_scaled)[0]
        prob_rf  = rf_model.predict_proba(feat_scaled)[0]
        prob_ens = 0.6 * prob_gb + 0.4 * prob_rf
        pred_class  = int(np.argmax(prob_ens))
        confidence  = float(prob_ens[pred_class])
        feat_dict   = dict(zip(feature_cols, feat_vec.tolist()))
        return jsonify({
            'success': True,
            'prediction': {
                'class': pred_class,
                'label': RISK_LABELS[pred_class],
                'color': RISK_COLORS[pred_class],
                'confidence': round(confidence * 100, 1),
                'probabilities': {
                    'Low':    round(float(prob_ens[0]) * 100, 1),
                    'Medium': round(float(prob_ens[1]) * 100, 1),
                    'High':   round(float(prob_ens[2]) * 100, 1),
                },
            },
            'features': {
                'means':  {k: round(feat_dict[f'mean_{k}'], 2)
                           for k in ['HR','MAP','SBP','Temp','O2Sat','Lactate','Creatinine','Platelets']},
                'trends': {k: round(feat_dict[f'trend_{k}'], 3)
                           for k in ['HR','MAP','SBP','Temp','O2Sat','Lactate','Creatinine','Platelets']},
                'shock_index': round(feat_dict['shock_index'], 3),
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/metrics')
def api_metrics():
    return jsonify(metrics)

@app.route('/api/hospitals')
def api_hospitals():
    try:
        lat = float(request.args.get('lat', ''))
        lng = float(request.args.get('lng', ''))
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid coordinates'}), 400

    cache_key = (round(lat, 3), round(lng, 3))
    cached = HOSPITAL_CACHE.get(cache_key)
    now = time.time()
    if cached and now - cached['timestamp'] < CACHE_TTL_SECONDS:
        return jsonify({
            'success': True,
            'elements': cached['elements'],
            'source': cached['source'],
            'cached': True,
        })

    query = (
        f'[out:json][timeout:25];('
        f'node["amenity"="hospital"](around:10000,{lat},{lng});'
        f'way["amenity"="hospital"](around:10000,{lat},{lng});'
        f'node["amenity"="clinic"](around:10000,{lat},{lng});'
        f'way["amenity"="clinic"](around:10000,{lat},{lng});'
        f'node["healthcare"="hospital"](around:10000,{lat},{lng});'
        f'way["healthcare"="hospital"](around:10000,{lat},{lng});'
        f');out center;'
    )
    payload = parse.urlencode({'data': query}).encode('utf-8')
    errors = []
    for endpoint in OVERPASS_ENDPOINTS:
        req = urlrequest.Request(
            endpoint,
            data=payload,
            headers={
                'Content-Type': 'application/x-www-form-urlencoded',
                'User-Agent': 'Rudransh-Sepsis-EWS/1.0',
            },
            method='POST',
        )

        try:
            with urlrequest.urlopen(req, timeout=8) as response:
                data = json.loads(response.read().decode('utf-8'))
            elements = data.get('elements', [])
            HOSPITAL_CACHE[cache_key] = {
                'timestamp': now,
                'elements': elements,
                'source': endpoint,
            }
            return jsonify({
                'success': True,
                'elements': elements,
                'source': endpoint,
                'cached': False,
            })
        except HTTPError as e:
            errors.append(f'{endpoint}: HTTP {e.code}')
        except URLError as e:
            errors.append(f'{endpoint}: {e.reason}')
        except Exception as e:
            errors.append(f'{endpoint}: {str(e)}')

    try:
        elements = nominatim_fallback(lat, lng)
        HOSPITAL_CACHE[cache_key] = {
            'timestamp': now,
            'elements': elements,
            'source': 'https://nominatim.openstreetmap.org/search',
        }
        return jsonify({
            'success': True,
            'elements': elements,
            'source': 'https://nominatim.openstreetmap.org/search',
            'cached': False,
            'fallback': True,
        })
    except Exception as e:
        errors.append(f'nominatim fallback: {str(e)}')

    print('Hospital lookup failed:', '; '.join(errors), flush=True)

    return jsonify({
        'success': False,
        'error': 'All hospital data providers failed',
        'detail': errors,
    }), 502

@app.route('/api/demo_case/<case_type>')
def demo_case(case_type):
    cases = {
        'low': {
            'HR':[72,74,71,73,75,72],'MAP':[92,90,91,93,91,90],
            'SBP':[122,120,118,121,119,120],'Temp':[36.8,36.9,37.0,36.9,36.8,37.0],
            'O2Sat':[98,97,98,98,97,98],'Lactate':[1.1,1.2,1.1,1.0,1.2,1.1],
            'Creatinine':[0.9,0.9,1.0,0.9,0.9,1.0],'Platelets':[230,228,232,229,231,227],
        },
        'medium': {
            'HR':[92,96,98,100,103,105],'MAP':[78,76,74,73,71,70],
            'SBP':[108,105,104,101,100,98],'Temp':[38.1,38.3,38.5,38.4,38.6,38.7],
            'O2Sat':[95,94,94,93,93,92],'Lactate':[2.1,2.3,2.5,2.6,2.8,3.0],
            'Creatinine':[1.4,1.5,1.6,1.7,1.8,1.9],'Platelets':[175,170,165,160,155,150],
        },
        'high': {
            'HR':[112,116,120,124,128,132],'MAP':[68,65,62,58,55,52],
            'SBP':[95,91,88,84,80,76],'Temp':[39.0,39.2,39.4,39.5,39.6,39.7],
            'O2Sat':[92,90,89,88,87,85],'Lactate':[3.8,4.2,4.6,5.0,5.5,6.0],
            'Creatinine':[2.2,2.5,2.8,3.1,3.4,3.8],'Platelets':[120,110,100,90,80,72],
        }
    }
    if case_type not in cases:
        return jsonify({'error': 'Invalid case'}), 400
    return jsonify({'vitals': cases[case_type]})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
