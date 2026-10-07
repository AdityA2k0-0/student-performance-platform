"""Student Performance Analytics Platform - FastAPI + scikit-learn demo."""
from datetime import datetime
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["study_hours", "attendance", "previous_marks", "assignment_score", "sleep_hours"]
LABELS = {
    "study_hours": "Study hours/day", "attendance": "Attendance %",
    "previous_marks": "Previous marks", "assignment_score": "Assignment score",
    "sleep_hours": "Sleep hours",
}

def grade(score: float) -> str:
    return "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D"

# ---------- synthetic data + training (runs once at startup) ----------
rng = np.random.default_rng(42)
N = 1000
df = pd.DataFrame({
    "study_hours": rng.uniform(0, 8, N),
    "attendance": rng.uniform(40, 100, N),
    "previous_marks": rng.uniform(30, 100, N),
    "assignment_score": rng.uniform(30, 100, N),
    "sleep_hours": rng.uniform(4, 9, N),
})
df["final_score"] = (
    0.35 * df.previous_marks + 0.20 * df.assignment_score + 0.18 * df.attendance
    + 3.0 * df.study_hours - 1.5 * np.abs(df.sleep_hours - 7.5) + rng.normal(0, 3, N)
).clip(0, 100)
df["grade"] = df.final_score.apply(grade)

X, y, yc = df[FEATURES], df["final_score"], df["grade"]
Xtr, Xte, ytr, yte, yctr, ycte = train_test_split(X, y, yc, test_size=0.2, random_state=1)
reg = RandomForestRegressor(n_estimators=150, random_state=1).fit(Xtr, ytr)
clf = RandomForestClassifier(n_estimators=150, random_state=1).fit(Xtr, yctr)
METRICS = {
    "r2": round(r2_score(yte, reg.predict(Xte)), 3),
    "mae": round(mean_absolute_error(yte, reg.predict(Xte)), 2),
    "accuracy": round(accuracy_score(ycte, clf.predict(Xte)), 3),
    "train_rows": len(Xtr), "test_rows": len(Xte), "algorithm": "Random Forest",
}
IMPORTANCE = sorted(
    [{"feature": LABELS[f], "importance": round(float(i), 3)} for f, i in zip(FEATURES, reg.feature_importances_)],
    key=lambda d: -d["importance"],
)
MEANS = X.mean()

# ---------- helpers ----------
def run_prediction(values: dict) -> dict:
    row = pd.DataFrame([values])[FEATURES]
    score = float(reg.predict(row)[0])
    probs = dict(zip(clf.classes_, clf.predict_proba(row)[0]))
    risk = round(float(probs.get("D", 0) + probs.get("C", 0) * 0.5), 2)
    # explanation: effect of replacing each feature with the dataset average
    effects = []
    for f in FEATURES:
        alt = row.copy(); alt[f] = MEANS[f]
        effects.append({"feature": LABELS[f], "impact": round(score - float(reg.predict(alt)[0]), 2)})
    effects.sort(key=lambda d: -abs(d["impact"]))
    tips = [
        f"Improve {e['feature'].lower()} - it is currently pulling your score down by {abs(e['impact'])} points."
        for e in effects if e["impact"] < -1
    ][:3] or ["Great work - keep your current routine going."]
    return {"score": round(score, 1), "grade": grade(score), "risk": risk,
            "at_risk": score < 50 or risk > 0.5, "explanation": effects, "suggestions": tips}

# ---------- API ----------
app = FastAPI(title="Student Performance Analytics Platform")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
HISTORY: list[dict] = []

class StudentInput(BaseModel):
    study_hours: float = Field(ge=0, le=12)
    attendance: float = Field(ge=0, le=100)
    previous_marks: float = Field(ge=0, le=100)
    assignment_score: float = Field(ge=0, le=100)
    sleep_hours: float = Field(ge=0, le=14)

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/predict")
def predict(data: StudentInput, save: bool = True):
    result = run_prediction(data.model_dump())
    if save:
        HISTORY.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "inputs": data.model_dump(),
                           "score": result["score"], "grade": result["grade"]})
        del HISTORY[20:]
    return result

@app.get("/api/history")
def history():
    return HISTORY

@app.get("/api/model-info")
def model_info():
    return {"metrics": METRICS, "importance": IMPORTANCE}

@app.get("/api/students")
def students():
    """Demo class roster with predictions for the teacher dashboard."""
    sample = df.sample(15, random_state=7).reset_index(drop=True)
    out = []
    for i, r in sample.iterrows():
        res = run_prediction({f: float(r[f]) for f in FEATURES})
        out.append({"id": i + 1, "name": f"Student {i + 1:02d}", "attendance": round(r.attendance),
                    "score": res["score"], "grade": res["grade"], "at_risk": res["at_risk"]})
    dist = {g: sum(1 for s in out if s["grade"] == g) for g in "ABCD"}
    return {"students": out, "distribution": dist,
            "average": round(sum(s["score"] for s in out) / len(out), 1)}
