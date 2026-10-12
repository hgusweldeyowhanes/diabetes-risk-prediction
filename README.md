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

## Deploy (Render API + Vercel frontend)

### Render

1. Create a Render Blueprint from this repository and review the services in `render.yaml`.
2. The blueprint creates a Python web service and a PostgreSQL database. The free database is temporary (Render deletes it after its free period); choose a persistent database plan before storing any real user data.
3. Set the web service's `FRONTEND_ORIGIN` to the exact Vercel origin, for example `https://your-app.vercel.app` (no trailing slash). Render generates `SECRET_KEY` and supplies `DATABASE_URL`.
4. The start command runs database migrations, trains/loads the model, then starts Gunicorn on Render's assigned port. The first model training can take a few minutes. Free web instances can sleep and have ephemeral filesystems, so model artifacts may need to be regenerated after restarts; use a suitable persistent storage/instance option or an intentionally managed model artifact for a production service.

### Vercel

1. Import this repository and set the project root to `frontend`.
2. Use `npm run build` as the build command and `dist` as the output directory. `frontend/vercel.json` enables client-side route fallback.
3. Add the `VITE_API_URL` environment variable with the Render API base URL, including `/api/v1`, for example `https://your-api.onrender.com/api/v1`. Redeploy after changing it; Vite embeds this value at build time.

Do not store real patient information in a free/demo deployment. Predictions are screening estimates, not a diagnosis.

## ML (advanced stack)

Calibrated stacking ensemble, SMOTE, iterative imputation, hybrid clinical rules, cost-sensitive weights, multi-tier risk, counterfactuals, ROC/calibration/PDP plots. Artifacts land in `backend/ml_artifacts/` and copies under `backend/static/models/`.
