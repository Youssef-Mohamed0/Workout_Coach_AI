"""FastAPI application serving the Workout Coach AI dashboard and biometric/pose inference."""

from contextlib import asynccontextmanager
import math
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.config import FEATURE_COLS, FRONTEND_INDEX, REAL_CSV_PATH
from backend.pose_analyzer import ModelNotLoadedError, SmartGymPoseAnalyzer
from backend.pose_math import EXERCISE_PROFILES, VALID_EXERCISES

MAX_BODY_BYTES = 65_536

FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="6" fill="#dc2626"/>'
    '<path d="M8 22V10h3.2l4.8 7 4.8-7H24v12h-3.2v-6.8L16.8 21h-1.6l-4-5.8V22H8z" fill="#fff"/>'
    "</svg>"
)


def _build_overview(analyzer: SmartGymPoseAnalyzer) -> dict[str, Any]:
    df = pd.read_csv(REAL_CSV_PATH)
    Xs = analyzer.scaler.transform(df[FEATURE_COLS])
    Xp = analyzer.pca.transform(Xs)
    clusters = analyzer.kmeans.predict(Xs)
    anom = analyzer.iso_forest.predict(Xs) == -1
    points = [
        {"pca1": round(float(p1), 3), "pca2": round(float(p2), 3), "cluster": int(c), "is_anomaly": bool(an)}
        for p1, p2, c, an in zip(Xp[:, 0], Xp[:, 1], clusters, anom)
    ]
    return {
        "total_records": int(len(df)),
        "pca_explained_variance": round(float(analyzer.pca.explained_variance_ratio_.sum() * 100), 2),
        "pca_explained_variance_2d": round(float(analyzer.pca.explained_variance_ratio_[:2].sum() * 100), 2),
        "total_anomalies": int(anom.sum()),
        "cluster_names": {
            str(k): {"en": v["en"], "ar": v["ar"], "size": v["size"]}
            for k, v in analyzer.cluster_profiles.items()
        },
        "metrics": analyzer.metrics_summary,
        "exercise_profiles": EXERCISE_PROFILES,
        "scatter_data": points,
    }


def _sanitize_json_value(value: Any) -> Any:
    """Replace non-finite floats and Exception objects so validation error responses serialize cleanly."""
    if isinstance(value, BaseException):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {k: _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(v) for v in value]
    return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.frontend_html = FRONTEND_INDEX.read_text(encoding="utf-8") if FRONTEND_INDEX.exists() else ""
    app.state.analyzer = None
    app.state.overview = None
    app.state.load_error = None
    try:
        analyzer = SmartGymPoseAnalyzer()
        app.state.analyzer = analyzer
        app.state.overview = _build_overview(analyzer)
    except ModelNotLoadedError as exc:
        app.state.load_error = str(exc)
    yield


app = FastAPI(title="SmartGym AI", lifespan=lifespan)

# Dashboard is served on the same local origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def security_and_size_guard(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH"):
        content_len = request.headers.get("content-length")
        if content_len is not None:
            try:
                if int(content_len) > MAX_BODY_BYTES:
                    return JSONResponse(status_code=413, content={"detail": "Request payload too large"})
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header"})
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return JSONResponse(status_code=413, content={"detail": "Request payload too large"})

    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    if request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.exception_handler(ModelNotLoadedError)
async def models_missing(request: Request, exc: ModelNotLoadedError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": _sanitize_json_value(exc.errors())})


def _get_analyzer() -> SmartGymPoseAnalyzer:
    analyzer = getattr(app.state, "analyzer", None)
    if analyzer is None:
        # Support lazy initialization when TestClient runs without lifespan context
        try:
            analyzer = SmartGymPoseAnalyzer()
            app.state.analyzer = analyzer
            app.state.overview = _build_overview(analyzer)
            app.state.load_error = None
        except ModelNotLoadedError as exc:
            app.state.load_error = str(exc)
            raise
    return analyzer


class BiometricPoseInput(BaseModel):
    """Strict input contract for biometric profile and live pose telemetry."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    Age: float = Field(ge=10, le=100)
    Weight_kg: float = Field(ge=30, le=250)
    Height_m: float = Field(ge=1.2, le=2.3)
    Session_Duration_min: float = Field(ge=1, le=300)

    Max_BPM: Optional[float] = Field(default=None, ge=80, le=230)
    Avg_BPM: Optional[float] = Field(default=None, ge=50, le=220)
    Resting_BPM: Optional[float] = Field(default=None, ge=30, le=120)
    Fat_Percentage: Optional[float] = Field(default=None, ge=2, le=60)
    Water_Intake_liters: Optional[float] = Field(default=None, ge=0, le=10)

    torso_lean_deg: float = Field(default=0.0, ge=0, le=90)
    symmetry_score: float = Field(default=100.0, ge=0, le=100)
    bad_reps: int = Field(default=0, ge=0, le=1000)
    exercise_type: str = Field(default="auto")

    @field_validator(
        "Age",
        "Weight_kg",
        "Height_m",
        "Session_Duration_min",
        "Max_BPM",
        "Avg_BPM",
        "Resting_BPM",
        "Fat_Percentage",
        "Water_Intake_liters",
        "torso_lean_deg",
        "symmetry_score",
        "bad_reps",
        mode="before",
    )
    @classmethod
    def _reject_booleans(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("Boolean values are not allowed for numeric fields")
        return value

    @field_validator("exercise_type")
    @classmethod
    def _valid_exercise(cls, value: str) -> str:
        if value not in VALID_EXERCISES:
            raise ValueError(f"Unsupported exercise_type '{value}'")
        return value

    @model_validator(mode="after")
    def _physiological_consistency(self):
        if self.Max_BPM is not None and self.Avg_BPM is not None and self.Avg_BPM > self.Max_BPM:
            raise ValueError("Avg_BPM cannot exceed Max_BPM")
        if self.Resting_BPM is not None and self.Avg_BPM is not None and self.Resting_BPM > self.Avg_BPM:
            raise ValueError("Resting_BPM cannot exceed Avg_BPM")
        if self.Resting_BPM is not None and self.Max_BPM is not None and self.Resting_BPM > self.Max_BPM:
            raise ValueError("Resting_BPM cannot exceed Max_BPM")
        bmi = self.Weight_kg / (self.Height_m ** 2)
        if not (10.0 <= bmi <= 65.0):
            raise ValueError(f"Computed BMI ({bmi:.1f}) must be between 10.0 and 65.0")
        return self


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    if getattr(app.state, "analyzer", None) is None:
        return JSONResponse(status_code=503, content={"status": "not_ready", "detail": getattr(app.state, "load_error", "Model not loaded")})
    return {"status": "ready"}


@app.get("/favicon.ico")
async def favicon():
    return Response(content=FAVICON_SVG, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/")
async def serve_dashboard():
    html = getattr(app.state, "frontend_html", None)
    if not html:
        html = FRONTEND_INDEX.read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/api/dataset-overview")
def get_dataset_overview():
    _get_analyzer()
    return app.state.overview


@app.post("/api/analyze-biometrics")
def analyze_biometrics(data: BiometricPoseInput):
    payload = data.model_dump()
    return {"ml_results": _get_analyzer().evaluate_real_biometrics_and_pose(payload)}
