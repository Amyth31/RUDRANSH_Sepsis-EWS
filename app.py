"""
SepsisAlert - Flask Backend
Routes: / (dashboard), /predict (prediction page), /map (hospital finder)
"""

from flask import Flask, request, jsonify, render_template
import pickle, json, os
import numpy as np
from urllib import parse, request as urlrequest
from urllib.error import HTTPError, URLError

app = Flask(__name__, template_folder='templates', static_folder='static')

BASE      = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, 'model')

with open(os.path.join(MODEL_DIR, 'gb_model.pkl'), 'rb') as f:
    gb_model = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'rf_model.pkl'), 'rb') as f:
    rf_model = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'feature_cols.json')) as f:
    feature_cols = json.load(f)
with open(os.path.join(MODEL_DIR, 'metrics.json')) as f:
    metrics = json.load(f)

RISK_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}
RISK_COLORS = {0: '#22c55e', 1: '#f59e0b', 2: '#ef4444'}

def compute_features(vitals_6h):
    keys = ['HR','MAP','SBP','Temp','O2Sat','Lactate','Creatinine','Platelets']
    feat = {}
    t = np.arange(6)
    for k in keys:
        s = np.array(vitals_6h[k], dtype=float)
        feat[f'mean_{k}']  = float(np.mean(s))
        feat[f'trend_{k}'] = float(np.polyfit(t, s, 1)[0])
    feat['shock_index'] = feat['mean_HR'] / max(feat['mean_SBP'], 1)
    return np.array([feat[c] for c in feature_cols])

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
    req = urlrequest.Request(
        'https://overpass-api.de/api/interpreter',
        data=payload,
        headers={
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': 'Rudransh-Sepsis-EWS/1.0',
        },
        method='POST',
    )

    try:
        with urlrequest.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
        return jsonify({'success': True, 'elements': data.get('elements', [])})
    except HTTPError as e:
        return jsonify({'success': False, 'error': f'Overpass HTTP {e.code}'}), 502
    except URLError as e:
        return jsonify({'success': False, 'error': f'Network error: {e.reason}'}), 502
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 502

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
