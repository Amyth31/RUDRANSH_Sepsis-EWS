# Rudransh — Sepsis Early Warning System
### AI/ML Semester Project & Research Paper
#### Designed for Indian Army Forward Field Deployment

---

## Mission

Rudransh is an AI-powered sepsis prediction system built specifically for Indian Army soldiers operating in forward field locations — Ladakh, Siachen, Arunachal Pradesh — where the nearest hospital can be 200+ km away and every hour of delayed treatment increases mortality by approximately 7%.

The system requires only equipment already present in every Regimental Aid Post (RAP): a pulse oximeter, BP cuff, and thermometer. It predicts sepsis risk up to **6 hours before clinical onset**, giving the Medical Officer time to initiate antibiotics, stabilize vitals, and arrange CASEVAC before the window closes.

---

## Project Structure

```
AIML_Project/
├── app.py                        # Flask REST API — all routes
├── train_model.py                # ML training pipeline (real PhysioNet data)
├── README.md                     # This file
│
├── model/                        # Auto-generated after training
│   ├── gb_model.pkl              # Gradient Boosting model
│   ├── rf_model.pkl              # Random Forest model
│   ├── scaler.pkl                # StandardScaler
│   ├── feature_cols.json         # Feature column order
│   └── metrics.json              # Performance metrics
│
├── templates/
│   ├── dashboard.html            # Main dashboard (scrolling, dark/light)
│   ├── index.html                # Predict page (patient login + hourly entry)
│   └── map.html                  # Hospital finder (Leaflet + CARTO dark tiles)
│
└── static/                       # Empty (all CSS/JS inline)
```

---

## Dataset

**PhysioNet Computing in Cardiology Challenge 2019**
- 40,336 ICU patients across 3 hospital systems
- Hourly measurements of 40+ physiological variables
- Sepsis-3 definition labeling

**Key Variables Used**

| Variable    | Description              | Normal Range     |
|-------------|--------------------------|------------------|
| HR          | Heart Rate               | 60–100 bpm       |
| MAP         | Mean Arterial Pressure   | 70–100 mmHg      |
| SBP         | Systolic Blood Pressure  | 90–120 mmHg      |
| Temp        | Temperature              | 36.5–37.5 °C     |
| O2Sat       | Oxygen Saturation        | 95–100 %         |
| Lactate     | Serum Lactate            | 0.5–1.5 mmol/L   |
| Creatinine  | Serum Creatinine         | 0.6–1.2 mg/dL    |
| Platelets   | Platelet Count           | 150–400 K/μL     |

---

## Feature Engineering — 17 Features

### 8 Mean Features
Average value of each vital over the 6-hour window:
`mean_HR`, `mean_MAP`, `mean_SBP`, `mean_Temp`, `mean_O2Sat`, `mean_Lactate`, `mean_Creatinine`, `mean_Platelets`

### 8 Trend Features
Linear regression slope (change per hour) over 6 hours:
`trend_HR`, `trend_MAP`, `trend_SBP`, `trend_Temp`, `trend_O2Sat`, `trend_Lactate`, `trend_Creatinine`, `trend_Platelets`

### 1 Shock Index
```
shock_index = mean_HR / mean_SBP
```
Values > 1.0 indicate hemodynamic shock risk.

---

## Model Architecture

### Ensemble: Gradient Boosting + Random Forest (60/40)

**Gradient Boosting Classifier**
- n_estimators: 400, max_depth: 5
- learning_rate: 0.08, subsample: 0.8
- Early stopping: 25 rounds patience

**Random Forest Classifier**
- n_estimators: 300, max_depth: 12
- Class-balanced weights

**Final Prediction**
```python
prob_ensemble = 0.6 * prob_GB + 0.4 * prob_RF
```

### Risk Classes
| Class | Label  | Criteria                              |
|-------|--------|---------------------------------------|
| 0     | Low    | No sepsis flag, Lactate < 2.0         |
| 1     | Medium | No sepsis flag, Lactate ≥ 2.0         |
| 2     | High   | SepsisLabel = 1 in any hour           |

---

## How to Run

### Prerequisites
```bash
pip install flask scikit-learn pandas numpy
```

### Step 1 — Set Dataset Paths in `train_model.py`
```python
DATASET_A = r"C:\path\to\training_setA"
DATASET_B = r"C:\path\to\training_setB"  # remove if only one set
```

### Step 2 — Train the Model
```bash
python train_model.py
```
This generates the `model/` folder automatically.

### Step 3 — Start the Server
```bash
python app.py
```

### Step 4 — Open in Browser
```
http://localhost:5000
```

---

## Pages

### Dashboard (`/`)
Scrolling minimal dashboard with:
- Mission statement and clinical context
- Recent patient records table (from localStorage DB)
- Global, India, and Forward Field sepsis impact
- Navigate section

### Predict (`/predict`)
Three-screen patient workflow:

**Screen 1 — Login**
- New Patient: Name, Age, Rank/Unit → auto-generates ID (e.g. `RDR-0001`)
- Existing Patient: Enter Patient ID to resume session
- Demo Cases: Pre-filled Stable / At-Risk / Critical examples

**Screen 2 — Hourly Entry**
- 6 individual hour slots (T-6h to T-0h)
- Enter each hour's vitals separately — no need to fill all at once
- Save each hour and return later using Patient ID
- Run Analysis unlocks after all 6 hours are filled

**Screen 3 — Result**
- Risk classification: Low / Medium / High
- Confidence score and probability bars
- Shock index, 6-hour waveform, engineered feature table
- Hourly vitals summary
- Result saved to dashboard patient DB

### Hospital Map (`/map`)
- CARTO dark tiles (switches to OpenStreetMap in light mode)
- Click "Use My Location" → GPS coordinates + 10km radius circle
- Real hospitals fetched from OpenStreetMap via Overpass API
- Search / filter by name
- Click hospital → highlight on map
- "Get Directions" → opens Google Maps with route

---

## UI Features

- **Dark / Light mode toggle** on every page (top right, next to clock)
- Theme persists across all pages via localStorage
- Scroll-reveal animations on dashboard
- Live clock in header

---

## API Reference

### `POST /api/predict`
```json
{
  "vitals": {
    "HR":         [72, 74, 71, 73, 75, 72],
    "MAP":        [92, 90, 91, 93, 91, 90],
    "SBP":        [122,120,118,121,119,120],
    "Temp":       [36.8,36.9,37.0,36.9,36.8,37.0],
    "O2Sat":      [98, 97, 98, 98, 97, 98],
    "Lactate":    [1.1, 1.2, 1.1, 1.0, 1.2, 1.1],
    "Creatinine": [0.9, 0.9, 1.0, 0.9, 0.9, 1.0],
    "Platelets":  [230,228,232,229,231,227]
  }
}
```

**Response**
```json
{
  "prediction": {
    "class": 0,
    "label": "Low",
    "confidence": 98.4,
    "probabilities": { "Low": 98.4, "Medium": 1.3, "High": 0.3 }
  },
  "features": {
    "means":  { "HR": 72.8, "MAP": 91.2, ... },
    "trends": { "HR": 0.03, "MAP": -0.2, ... },
    "shock_index": 0.61
  }
}
```

### `GET /api/demo_case/{low|medium|high}`
Returns pre-built demo vitals for testing.

### `GET /api/metrics`
Returns model performance metrics from `model/metrics.json`.

---

## Tech Stack

| Layer      | Technology                                      |
|------------|-------------------------------------------------|
| ML         | scikit-learn (GradientBoosting + RandomForest)  |
| Data       | NumPy, Pandas                                   |
| Backend    | Python 3, Flask                                 |
| Frontend   | HTML5, CSS3, Vanilla JavaScript                 |
| Map        | Leaflet.js + CARTO dark tiles + Overpass API    |
| Fonts      | Syne (headings), Space Mono (data/mono)         |
| Storage    | localStorage (patient DB, theme preference)     |

---

## Clinical Context — Forward Field Application

In India's forward military zones (Siachen Glacier, Eastern Ladakh, Arunachal border), soldiers face:

- **Extreme cold** — suppresses early immune response, masks fever
- **Altitude hypoxia** — accelerates multi-organ dysfunction
- **Combat trauma** — direct infection pathway into bloodstream
- **Zero ICU access** — nearest ECHS facility often 200+ km away

Rudransh addresses this by:
1. Using only vitals measurable with RAP-standard equipment
2. Providing a 6-hour early warning before sepsis becomes irreversible
3. Giving a clear Low / Medium / High classification to guide CASEVAC decisions
4. Running on any laptop or tablet — no internet required for prediction

---

## Sepsis Impact Data

| Scope          | Statistic                                    |
|----------------|----------------------------------------------|
| Global cases   | 49 million per year                          |
| Global deaths  | 11 million per year (1 in 5 all deaths)      |
| India cases    | 11.3 million per year                        |
| India deaths   | 3.3 million per year                         |
| ICU mortality  | 50–70% in resource-limited settings          |
| Delay impact   | ~7% mortality increase per hour untreated    |

---

## References

1. Reyna et al. (2020). *Early Prediction of Sepsis From Clinical Data — The PhysioNet/Computing in Cardiology Challenge 2019.* Critical Care Medicine.
2. Singer et al. (2016). *The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).* JAMA.
3. Rudd et al. (2020). *Global, regional, and national sepsis incidence and mortality.* The Lancet.
4. ICMR Sepsis Guidelines (2020). *Indian Council of Medical Research.*
5. Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.

---

*Rudransh — Built for soldiers who cannot wait for a hospital.*
*Indian Army Medical Corps · Forward Field Deployment · CASEVAC Decision Support*
