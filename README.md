# Diabetes Risk Prediction

Modular full-stack app: Django API + advanced ML + React UI.

## Layout

```
diabetes-risk-prediction/
├── backend/                 # Django
│   ├── config/              # settings, urls, wsgi, asgi
│   ├── core/                # health
│   ├── accounts/            # register, JWT, profiles
│   ├── patients/            # patient records
│   ├── predictions/         # predict API, history, model metadata
│   ├── ml/                  # pipeline, preprocessor, explainer, validator
│   ├── api/v1/              # versioned routes
│   ├── utils/               # middleware, helpers
│   ├── static/models/       # exported joblib artifacts
│   ├── data/diabetes.csv
│   └── tests/
├── frontend/                # React (Vite)
│   ├── src/components/
│   ├── src/pages/
│   └── src/services/api.js
├── docker-compose.yml
└── README.md
```

## Backend

```bash
cd backend
# use project venv:  source ../.venv/Scripts/activate   (or create backend/.venv)
pip install -r requirements.txt
cp .env.example .env   # optional
python manage.py migrate
python manage.py train_model --force   # ~1–2 min first time
python manage.py runserver
```

API base: `http://127.0.0.1:8000/api/v1/`

| Method | Path | Notes |
|--------|------|--------|
| GET | `/health/` | Service + model status |
| POST | `/auth/register/` | Create user |
| POST | `/auth/token/` | JWT pair |
| POST | `/predictions/run/` | Biomarker risk score (public) |
| GET | `/predictions/history/` | Auth user history |
| CRUD | `/patients/` | Auth patient registry |
| GET | `/predictions/models/active/` | Active model metrics |

Predict body:

```json
{
  "pregnancies": 1,
  "glucose": 148,
  "blood_pressure": 72,
  "skin_thickness": 35,
  "insulin": 0,
  "bmi": 33.6,
  "diabetes_pedigree_function": 0.627,
  "age": 50
}
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — Vite proxies `/api` to the Django server when configured.

## Docker

```bash
docker compose up --build
```

Spins Postgres + backend (migrates + trains) + frontend.

## ML (advanced stack)

Calibrated stacking ensemble, SMOTE, iterative imputation, hybrid clinical rules, cost-sensitive weights, multi-tier risk, counterfactuals, ROC/calibration/PDP plots. Artifacts land in `backend/ml_artifacts/` and copies under `backend/static/models/`.
