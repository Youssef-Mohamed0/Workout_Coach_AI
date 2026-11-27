"""Biomechanical movement risk, anomaly thresholds, and shared exercise profiles."""

import math
from typing import Any

# Default movement safety thresholds (shared by backend and frontend)
LEAN_LIMIT_DEG = 17.5
SYMMETRY_FLOOR = 82.0
LEAN_ANOMALY_DEG = 20.0
SYMMETRY_ANOMALY = 76.0
BAD_REPS_ANOMALY = 2

# Unified per-exercise biomechanics configuration (Single Source of Truth)
EXERCISE_PROFILES: dict[str, dict[str, Any]] = {
    "auto": {
        "title": "Full Body Smart Auto-Detection (Arms, Shoulders, Back & Legs)",
        "mode": "reps",
        "direction": "flex",
        "jointsL": [11, 13, 15],
        "jointsR": [12, 14, 16],
        "startThreshold": 138.0,
        "peakThreshold": 72.0,
        "minRom": 35.0,
        "maxAllowedLean": LEAN_LIMIT_DEG,
        "leanAnomalyDeg": LEAN_ANOMALY_DEG,
        "symmetryFloor": SYMMETRY_FLOOR,
        "symmetryAnomaly": SYMMETRY_ANOMALY,
    },
    "bicep_curl": {
        "title": "Bicep / Hammer Curl — Shoulder -> Elbow -> Wrist",
        "mode": "reps",
        "direction": "flex",
        "jointsL": [11, 13, 15],
        "jointsR": [12, 14, 16],
        "startThreshold": 138.0,
        "peakThreshold": 72.0,
        "minRom": 35.0,
        "maxAllowedLean": LEAN_LIMIT_DEG,
        "leanAnomalyDeg": LEAN_ANOMALY_DEG,
        "symmetryFloor": SYMMETRY_FLOOR,
        "symmetryAnomaly": SYMMETRY_ANOMALY,
    },
    "shoulder_press": {
        "title": "Overhead Shoulder Press — Hip -> Shoulder -> Elbow",
        "mode": "reps",
        "direction": "extend",
        "jointsL": [13, 11, 23],
        "jointsR": [14, 12, 24],
        "startThreshold": 85.0,
        "peakThreshold": 145.0,
        "minRom": 35.0,
        "maxAllowedLean": 18.0,
        "leanAnomalyDeg": 22.0,
        "symmetryFloor": SYMMETRY_FLOOR,
        "symmetryAnomaly": SYMMETRY_ANOMALY,
    },
    "lateral_raise": {
        "title": "Side Lateral Raise — Hip -> Shoulder -> Elbow Abduction",
        "mode": "reps",
        "direction": "extend",
        "jointsL": [23, 11, 13],
        "jointsR": [24, 12, 14],
        "startThreshold": 30.0,
        "peakThreshold": 75.0,
        "minRom": 30.0,
        "maxAllowedLean": 16.0,
        "leanAnomalyDeg": 20.0,
        "symmetryFloor": SYMMETRY_FLOOR,
        "symmetryAnomaly": SYMMETRY_ANOMALY,
    },
    "pushup_bench": {
        "title": "Push-Up / Bench Press — Shoulder -> Elbow -> Wrist",
        "mode": "reps",
        "direction": "flex",
        "jointsL": [11, 13, 15],
        "jointsR": [12, 14, 16],
        "startThreshold": 140.0,
        "peakThreshold": 85.0,
        "minRom": 35.0,
        "leanMetric": "body_line",
        "maxAllowedLean": 20.0,
        "leanAnomalyDeg": 28.0,
        "symmetryFloor": SYMMETRY_FLOOR,
        "symmetryAnomaly": SYMMETRY_ANOMALY,
    },
    "lat_row": {
        "title": "Lat Pulldown / Back Row — Shoulder -> Elbow -> Wrist",
        "mode": "reps",
        "direction": "flex",
        "jointsL": [11, 13, 15],
        "jointsR": [12, 14, 16],
        "startThreshold": 140.0,
        "peakThreshold": 75.0,
        "minRom": 35.0,
        "maxAllowedLean": 20.0,
        "leanAnomalyDeg": 25.0,
        "symmetryFloor": SYMMETRY_FLOOR,
        "symmetryAnomaly": SYMMETRY_ANOMALY,
    },
    "squat": {
        "title": "Squat — Hip -> Knee -> Ankle Depth & Alignment",
        "mode": "reps",
        "direction": "flex",
        "jointsL": [23, 25, 27],
        "jointsR": [24, 26, 28],
        "startThreshold": 145.0,
        "peakThreshold": 95.0,
        "minRom": 35.0,
        "maxAllowedLean": 25.0,
        "leanAnomalyDeg": 32.0,
        "symmetryFloor": SYMMETRY_FLOOR,
        "symmetryAnomaly": SYMMETRY_ANOMALY,
    },
    "deadlift": {
        "title": "Deadlift — Shoulder -> Hip -> Knee Hinge",
        "mode": "reps",
        "direction": "flex",
        "jointsL": [11, 23, 25],
        "jointsR": [12, 24, 26],
        "startThreshold": 150.0,
        "peakThreshold": 100.0,
        "minRom": 35.0,
        "maxAllowedLean": 45.0,
        "leanAnomalyDeg": 55.0,
        "symmetryFloor": SYMMETRY_FLOOR,
        "symmetryAnomaly": SYMMETRY_ANOMALY,
    },
    "lunge": {
        "title": "Lunges / Split Squat — Knee Flexion & Pelvic Balance",
        "mode": "reps",
        "direction": "flex",
        "jointsL": [23, 25, 27],
        "jointsR": [24, 26, 28],
        "startThreshold": 145.0,
        "peakThreshold": 92.0,
        "minRom": 35.0,
        "maxAllowedLean": 20.0,
        "leanAnomalyDeg": 25.0,
        "symmetryFloor": 78.0,
        "symmetryAnomaly": 70.0,
    },
    "plank": {
        "title": "Plank / Core Stability — Shoulder -> Hip -> Ankle Hold Timer",
        "mode": "hold",
        "direction": "hold",
        "jointsL": [11, 23, 27],
        "jointsR": [12, 24, 28],
        "minHoldAngle": 155.0,
        "leanMetric": "body_line",
        "maxAllowedLean": 22.0,
        "leanAnomalyDeg": 30.0,
        "symmetryFloor": SYMMETRY_FLOOR,
        "symmetryAnomaly": SYMMETRY_ANOMALY,
    },
}

VALID_EXERCISES = frozenset(EXERCISE_PROFILES.keys())


def get_exercise_profile(exercise_type: str = "auto") -> dict[str, Any]:
    """Return the biomechanics threshold profile for an exercise mode."""
    return EXERCISE_PROFILES.get(exercise_type, EXERCISE_PROFILES["auto"])


def movement_risk(
    torso_lean: float,
    symmetry: float,
    bad_reps: int,
    exercise_type: str = "auto",
) -> tuple[float, list[str]]:
    """Compute movement risk (0-100) and diagnostic flags using exercise-aware limits."""
    if not (math.isfinite(torso_lean) and math.isfinite(symmetry)):
        raise ValueError("torso_lean and symmetry must be finite numbers")
    if bad_reps < 0:
        raise ValueError("bad_reps cannot be negative")

    profile = get_exercise_profile(exercise_type)
    lean_limit = float(profile["maxAllowedLean"])
    sym_floor = float(profile["symmetryFloor"])
    lean_label = "body-line deviation" if profile.get("leanMetric") == "body_line" else "torso lean"

    risk, flags = 0.0, []
    if torso_lean > lean_limit:
        risk += (torso_lean - lean_limit) * 2.8
        flags.append(f"{lean_label} {torso_lean:.1f} deg > {lean_limit}")
    if symmetry < sym_floor:
        risk += (sym_floor - symmetry) * 1.9
        flags.append(f"left/right symmetry {symmetry:.1f}% < {sym_floor}")
    if bad_reps > 0:
        risk += min(45.0, bad_reps * 12.0)
        flags.append(f"{bad_reps} bad rep(s)")
    return round(min(100.0, risk), 1), flags


def movement_anomaly(
    torso_lean: float,
    symmetry: float,
    bad_reps: int,
    exercise_type: str = "auto",
) -> bool:
    """Return True if posture deviation, asymmetry, or repeated bad reps indicate anomaly."""
    if not (math.isfinite(torso_lean) and math.isfinite(symmetry)):
        raise ValueError("torso_lean and symmetry must be finite numbers")

    profile = get_exercise_profile(exercise_type)
    lean_anom = float(profile["leanAnomalyDeg"])
    sym_anom = float(profile["symmetryAnomaly"])
    return bool(
        torso_lean > lean_anom
        or symmetry < sym_anom
        or bad_reps >= BAD_REPS_ANOMALY
    )
