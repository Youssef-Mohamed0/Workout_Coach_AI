"""Runtime biometric and pose fusion engine wrapping the frozen ML artifacts."""

from functools import lru_cache
import math
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from backend.config import FEATURE_COLS, MODELS_PATH, PHYSIO_KEYS, REAL_CSV_PATH
from backend.features import enhance_features
from backend.pose_math import movement_anomaly, movement_risk

REQUIRED = ("Age", "Weight_kg", "Height_m", "Session_Duration_min")
UNIVARIATE_Z_THRESHOLD = 3.5
CLINICAL_RESTING_HR_HIGH = 100.0
CLINICAL_AVG_HR_PCT_HIGH = 0.90
CLINICAL_BMI_MIN = 12.0
CLINICAL_BMI_MAX = 45.0


class ModelNotLoadedError(RuntimeError):
    """Raised when the trained model bundle is missing or cannot be loaded."""


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def physio_rule_flags(
    age: float,
    max_bpm: Optional[float],
    avg_bpm: Optional[float],
    resting_bpm: Optional[float],
    water_l: Optional[float],
    duration_min: float,
) -> list[str]:
    """Evaluate deterministic cardiac/hydration rules on whichever inputs were provided."""
    if not (_is_finite_number(age) and _is_finite_number(duration_min)):
        raise ValueError("Age and Session_Duration_min must be finite numbers")

    hr_max_pred = 220.0 - float(age)
    if hr_max_pred <= 0.0:
        raise ValueError("Age must be below 220 to estimate predicted max heart rate")

    flags: list[str] = []
    if max_bpm is not None and float(max_bpm) / hr_max_pred > 0.96:
        flags.append("max HR above 96% of age-predicted maximum")

    if avg_bpm is not None:
        avg_val = float(avg_bpm)
        if water_l is not None and avg_val > 155.0 and float(water_l) < 2.0:
            flags.append("high average HR with low water intake")
        elif avg_val / hr_max_pred > CLINICAL_AVG_HR_PCT_HIGH:
            flags.append("sustained average HR above 90% of age-predicted maximum")

    if resting_bpm is not None:
        rest_val = float(resting_bpm)
        if rest_val > 82.0 and float(duration_min) > 85.0:
            flags.append("high resting HR with a long session")
        elif rest_val > CLINICAL_RESTING_HR_HIGH:
            flags.append("elevated resting HR above 100 BPM")

    return flags


class SmartGymPoseAnalyzer:
    """Loads frozen scikit-learn artifacts and evaluates biometric + pose payloads."""

    def __init__(self, models_path=MODELS_PATH):
        if not models_path.exists():
            raise ModelNotLoadedError("ML model bundle is missing. Run: python -m backend.ml_pipeline")
        a = joblib.load(models_path)
        self.scaler, self.pca, self.kmeans = a["scaler"], a["pca"], a["kmeans"]
        self.iso_forest, self.classifier, self.regressor = a["iso_forest"], a["classifier"], a["regressor"]
        self.feature_cols = a["feature_cols"]
        self.metrics_summary = a.get("metrics_summary", {})
        self.cluster_profiles = a.get("cluster_profiles", {})
        self.medians = a.get("training_medians") or pd.read_csv(REAL_CSV_PATH)[FEATURE_COLS].median().to_dict()
        self._scaler_stats = {
            col: (float(mean), float(scale))
            for col, mean, scale in zip(self.feature_cols, self.scaler.mean_, self.scaler.scale_)
        }

    def cluster_label(self, k: int) -> str:
        c = self.cluster_profiles.get(k)
        return f"Cluster {k}: {c['en']} ({c['ar']})" if c else f"Cluster {k}"

    def _univariate_clinical_anomalies(self, observed: dict[str, float]) -> list[str]:
        """Flag extreme univariate z-scores or out-of-bounds clinical values on non-imputed inputs."""
        reasons: list[str] = []
        for col, val in observed.items():
            mean, scale = self._scaler_stats.get(col, (0.0, 1.0))
            if scale > 0.0:
                z = (val - mean) / scale
                if abs(z) > UNIVARIATE_Z_THRESHOLD:
                    reasons.append(f"{col} extreme outlier (z={z:+.2f})")
        bmi = observed.get("BMI")
        if bmi is not None and (bmi < CLINICAL_BMI_MIN or bmi > CLINICAL_BMI_MAX):
            if not any(r.startswith("BMI ") for r in reasons):
                reasons.append(f"BMI outside typical clinical range ({bmi:.1f})")
        return reasons

    @lru_cache(maxsize=256)
    def _predict_biometrics_cached(
        self,
        age: float,
        weight: float,
        height: float,
        dur: float,
        max_b: float,
        avg_b: float,
        rest_b: float,
        water: float,
        fat: float,
        bmi_rounded: float,
    ) -> tuple[tuple[float, float, float], int, float, float, bool]:
        """Memoize model inference on the 10-column biometric row to keep live frame requests fast."""
        row = pd.DataFrame(
            [
                {
                    "Age": age,
                    "Weight_kg": weight,
                    "Height_m": height,
                    "Max_BPM": max_b,
                    "Avg_BPM": avg_b,
                    "Resting_BPM": rest_b,
                    "Session_Duration_min": dur,
                    "Fat_Percentage": fat,
                    "Water_Intake_liters": water,
                    "BMI": bmi_rounded,
                }
            ]
        )[self.feature_cols]

        enhanced, Xp, Xs = enhance_features(row, self.scaler, self.pca, self.kmeans)
        cluster_id = int(enhanced["cluster_id"].iloc[0])
        calories = float(self.regressor.predict(enhanced)[0])
        clf_prob = float(self.classifier.predict_proba(enhanced)[0, 1]) * 100.0
        iso_anom = bool(self.iso_forest.predict(Xs)[0] == -1)
        pca_coords = (
            round(float(Xp[0, 0]), 3),
            round(float(Xp[0, 1]), 3),
            round(float(Xp[0, 2]), 3),
        )
        return pca_coords, cluster_id, round(calories, 1), round(clf_prob, 1), iso_anom

    def evaluate_real_biometrics_and_pose(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate inputs, run biometric + pose evaluation, and fuse safety verdicts."""
        imputed: list[str] = []
        raw_optional: dict[str, Optional[float]] = {}

        def get_required(key: str) -> float:
            v = data.get(key)
            if v is None:
                raise ValueError(f"{key} is required")
            if not _is_finite_number(v):
                raise ValueError(f"{key} must be a finite number")
            return float(v)

        def get_optional(key: str) -> float:
            v = data.get(key)
            if v is None:
                raw_optional[key] = None
                imputed.append(key)
                return float(self.medians[key])
            if not _is_finite_number(v):
                raise ValueError(f"{key} must be a finite number")
            val = float(v)
            raw_optional[key] = val
            return val

        age, weight, height, dur = (get_required(k) for k in REQUIRED)
        if not (10.0 <= age <= 100.0):
            raise ValueError("Age must be between 10 and 100")
        if not (30.0 <= weight <= 250.0):
            raise ValueError("Weight_kg must be between 30 and 250")
        if not (1.2 <= height <= 2.3):
            raise ValueError("Height_m must be between 1.2 and 2.3")
        if not (1.0 <= dur <= 300.0):
            raise ValueError("Session_Duration_min must be between 1 and 300")

        bmi_exact = weight / (height ** 2)
        if not (10.0 <= bmi_exact <= 65.0):
            raise ValueError("Computed BMI must be between 10.0 and 65.0")

        max_b, avg_b, rest_b, water, fat = (
            get_optional(k)
            for k in ("Max_BPM", "Avg_BPM", "Resting_BPM", "Water_Intake_liters", "Fat_Percentage")
        )

        raw_max, raw_avg, raw_rest, raw_water = (
            raw_optional["Max_BPM"],
            raw_optional["Avg_BPM"],
            raw_optional["Resting_BPM"],
            raw_optional["Water_Intake_liters"],
        )
        if raw_avg is not None and raw_max is not None and raw_avg > raw_max:
            raise ValueError("Avg_BPM cannot exceed Max_BPM")
        if raw_rest is not None and raw_avg is not None and raw_rest > raw_avg:
            raise ValueError("Resting_BPM cannot exceed Avg_BPM")
        if raw_rest is not None and raw_max is not None and raw_rest > raw_max:
            raise ValueError("Resting_BPM cannot exceed Max_BPM")

        pca_coords, cluster_id, calories, clf_prob, iso_anom = self._predict_biometrics_cached(
            age, weight, height, dur, max_b, avg_b, rest_b, water, fat, round(bmi_exact, 2)
        )

        physio_reliable = not any(k in imputed for k in PHYSIO_KEYS)
        physio_risk: Optional[float] = clf_prob if physio_reliable else None
        rule_flags = physio_rule_flags(age, raw_max, raw_avg, raw_rest, raw_water, dur)

        observed_for_guard: dict[str, float] = {
            "Age": age,
            "Weight_kg": weight,
            "Height_m": height,
            "Session_Duration_min": dur,
            "BMI": round(bmi_exact, 2),
        }
        for key, raw_val in raw_optional.items():
            if raw_val is not None:
                observed_for_guard[key] = raw_val

        clinical_anomalies = self._univariate_clinical_anomalies(observed_for_guard)
        bio_anomaly = bool((physio_reliable and iso_anom) or clinical_anomalies)

        lean_raw = data.get("torso_lean_deg", 0.0)
        sym_raw = data.get("symmetry_score", 100.0)
        bad_raw = data.get("bad_reps", 0)
        if not (_is_finite_number(lean_raw) and _is_finite_number(sym_raw)):
            raise ValueError("torso_lean_deg and symmetry_score must be finite numbers")
        if isinstance(bad_raw, bool) or not isinstance(bad_raw, (int, float)) or not math.isfinite(float(bad_raw)):
            raise ValueError("bad_reps must be a finite integer")

        lean = float(lean_raw)
        sym = float(sym_raw)
        bad = int(bad_raw)
        exercise_type = str(data.get("exercise_type", "auto"))

        mv_risk, mv_flags = movement_risk(lean, sym, bad, exercise_type=exercise_type)
        mv_anom = movement_anomaly(lean, sym, bad, exercise_type=exercise_type)
        is_anom = bool(bio_anomaly or mv_anom)

        # Ensure deterministic rule flags or clinical/statistical anomalies raise the risk floor above safe
        clinical_floor = 0.0
        if rule_flags:
            clinical_floor = max(clinical_floor, min(95.0, 55.0 + 12.0 * (len(rule_flags) - 1)))
        if bio_anomaly:
            clinical_floor = max(clinical_floor, 55.0)
        if mv_anom:
            clinical_floor = max(clinical_floor, 50.0)

        effective_physio = max(physio_risk or 0.0, clinical_floor) if (rule_flags or bio_anomaly) else (physio_risk or 0.0)
        overall = round(float(np.clip(max(effective_physio, mv_risk), 0.0, 100.0)), 1)

        return {
            "pca_coordinates": list(pca_coords),
            "cluster_id": cluster_id,
            "cluster_profile": self.cluster_label(cluster_id),
            "predicted_calories_kcal": calories,
            "physio_risk_pct": physio_risk,
            "physio_rule_flags": rule_flags,
            "clinical_anomaly_flags": clinical_anomalies,
            "movement_risk_pct": mv_risk,
            "movement_quality_score": round(100.0 - mv_risk, 1),
            "movement_flags": mv_flags,
            "imputed_fields": imputed,
            "injury_risk_probability": overall,
            "injury_risk_label": "High Injury / Overtraining Risk" if overall >= 50.0 else "Safe & Optimal Performance",
            "is_anomaly": is_anom,
            "anomaly_status": "DANGER: Biometric / Pose Anomaly Flagged" if is_anom else "Normal Pattern",
        }
