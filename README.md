# Student Performance Analytics Platform (demo)

FastAPI + scikit-learn backend, React (Vite) frontend.

## Run (two terminals)
**Backend**
```
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```
API docs: http://localhost:8000/docs

**Frontend**
```
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

## What it does
- **Predict**: sliders for 5 factors -> predicted score, grade, at-risk flag, per-factor explanation, suggestions (live what-if)
- **Teacher Dashboard**: demo class roster, at-risk detection, grade distribution
- **Model Info**: R2, MAE, accuracy, feature importance

## How it works (viva notes)
- Synthetic dataset (1000 rows) generated at startup with realistic relationships
- Random Forest Regressor predicts score; Random Forest Classifier predicts grade band
- Explanation = change in prediction when a feature is replaced by the dataset average
- Pydantic validates input; CORS enabled; Vite proxies /api to FastAPI

## Next steps (later)
JWT auth + roles, SQLAlchemy database, CSV upload, PDF reports, model retraining, Docker, pytest tests.
