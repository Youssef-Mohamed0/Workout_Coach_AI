from functools import lru_cache
from typing import Optional

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator

from backend.config import FEATURE_COLS, FRONTEND_INDEX, REAL_CSV_PATH
from backend.pose_analyzer import SmartGymPoseAnalyzer

app = FastAPI(title="SmartGym AI")

# the page is served by this same server, so only local addresses are allowed
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"], allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


# if the models file is missing, say what to do instead of a plain 500 error
@app.exception_handler(FileNotFoundError)
def models_missing(request, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# loads the models the first time, then reuses them
@lru_cache(maxsize=1)
def get_analyzer() -> SmartGymPoseAnalyzer:
    return SmartGymPoseAnalyzer()


# the data coming from the page, with limits on every number
class BiometricPoseInput(BaseModel):
    # body data (required)
    Age: float = Field(ge=10, le=100)
    Weight_kg: float = Field(ge=30, le=250)
    Height_m: float = Field(ge=1.2, le=2.3)
    Session_Duration_min: float = Field(ge=1, le=300)
    # smartwatch data (optional)
    Max_BPM: Optional[float] = Field(default=None, ge=80, le=230)
    Avg_BPM: Optional[float] = Field(default=None, ge=50, le=220)
    Resting_BPM: Optional[float] = Field(default=None, ge=30, le=120)
    Fat_Percentage: Optional[float] = Field(default=None, ge=2, le=60)
    Water_Intake_liters: Optional[float] = Field(default=None, ge=0, le=10)
    # numbers from the camera (MediaPipe)
    torso_lean_deg: float = Field(default=0.0, ge=0, le=90)
    symmetry_score: float = Field(default=100.0, ge=0, le=100)
    bad_reps: int = Field(default=0, ge=0, le=1000)

    # avg heart rate can not be higher than max heart rate
    @model_validator(mode="after")
    def _hr_consistent(self):
        if self.Max_BPM is not None and self.Avg_BPM is not None and self.Avg_BPM > self.Max_BPM:
            raise ValueError("Avg_BPM cannot exceed Max_BPM")
        return self


# opens the page
@app.get("/")
def serve_dashboard():
    return FileResponse(FRONTEND_INDEX)


# data for the scatter plot, calculated once and then reused
@lru_cache(maxsize=1)
def _overview() -> dict:
    a = get_analyzer()
    df = pd.read_csv(REAL_CSV_PATH)
    Xs = a.scaler.transform(df[FEATURE_COLS])
    Xp = a.pca.transform(Xs)
    clusters = a.kmeans.predict(Xs)
    anom = a.iso_forest.predict(Xs) == -1
    points = [{"pca1": round(float(p1), 3), "pca2": round(float(p2), 3), "cluster": int(c),
               "is_anomaly": bool(an), "calories": float(cal)}
              for p1, p2, c, an, cal in zip(Xp[:, 0], Xp[:, 1], clusters, anom, df["Calories_Burned"])]
    return {
        "total_records": int(len(df)),
        "pca_explained_variance": round(float(a.pca.explained_variance_ratio_.sum() * 100), 2),
        "pca_explained_variance_2d": round(float(a.pca.explained_variance_ratio_[:2].sum() * 100), 2),
        "total_anomalies": int(anom.sum()),
        "cluster_names": {str(k): {"en": v["en"], "ar": v["ar"], "size": v["size"]} for k, v in a.cluster_profiles.items()},
        "metrics": a.metrics_summary,
        "scatter_data": points,
    }


# the page calls this when it opens, to draw the map of all members
@app.get("/api/dataset-overview")
def get_dataset_overview():
    return _overview()


# the page calls this every half second or so while the video plays
@app.post("/api/analyze-biometrics")
def analyze_biometrics(data: BiometricPoseInput):
    payload = data.model_dump()
    return {"ml_results": get_analyzer().evaluate_real_biometrics_and_pose(payload)}
