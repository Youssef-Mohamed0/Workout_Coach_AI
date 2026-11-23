import joblib
import pandas as pd

from backend.config import FEATURE_COLS, MODELS_PATH, PHYSIO_KEYS, REAL_CSV_PATH
from backend.features import enhance_features
from backend.pose_math import movement_anomaly, movement_risk

# these 4 must always be sent, the rest are optional
REQUIRED = ("Age", "Weight_kg", "Height_m", "Session_Duration_min")


# the reasons why the heart/hydration rule says "risk" (empty list if no rule fired)
def physio_rule_flags(age, max_bpm, avg_bpm, resting_bpm, water_l, duration_min):
    flags = []
    if max_bpm / (220.0 - age) > 0.96:
        flags.append("max HR above 96% of age-predicted maximum")
    if avg_bpm > 155 and water_l < 2.0:
        flags.append("high average HR with low water intake")
    if resting_bpm > 82 and duration_min > 85:
        flags.append("high resting HR with a long session")
    return flags


class SmartGymPoseAnalyzer:
    # loads the saved models once when the server starts
    def __init__(self, models_path=MODELS_PATH):
        if not models_path.exists():
            raise FileNotFoundError(f"{models_path} not found. Run: python -m backend.ml_pipeline")
        a = joblib.load(models_path)
        self.scaler, self.pca, self.kmeans = a["scaler"], a["pca"], a["kmeans"]
        self.iso_forest, self.classifier, self.regressor = a["iso_forest"], a["classifier"], a["regressor"]
        self.feature_cols = a["feature_cols"]
        self.metrics_summary = a.get("metrics_summary", {})
        self.cluster_profiles = a.get("cluster_profiles", {})
        self.medians = a.get("training_medians") or pd.read_csv(REAL_CSV_PATH)[FEATURE_COLS].median().to_dict()

    # full name of a cluster, like "Cluster 0: high weight, high BMI (...)"
    def cluster_label(self, k):
        c = self.cluster_profiles.get(k)
        return f"Cluster {k}: {c['en']} ({c['ar']})" if c else f"Cluster {k}"

    # takes the user data + the numbers from the camera, returns the full result
    def evaluate_real_biometrics_and_pose(self, data):
        imputed = []   # optional fields the user did not send (filled with the training median)

        # reads one value, fills it with the median if it is optional and missing
        def get(key):
            v = data.get(key)
            if v is None:
                if key in REQUIRED:
                    raise ValueError(f"{key} is required")
                imputed.append(key)
                return float(self.medians[key])
            return float(v)

        age, weight, height, dur = (get(k) for k in REQUIRED)
        max_b, avg_b, rest_b, water, fat = (get(k) for k in
            ("Max_BPM", "Avg_BPM", "Resting_BPM", "Water_Intake_liters", "Fat_Percentage"))
        bmi = weight / (height ** 2)

        # one row with the 10 columns in the same order used in training
        row = pd.DataFrame([{"Age": age, "Weight_kg": weight, "Height_m": height, "Max_BPM": max_b,
                             "Avg_BPM": avg_b, "Resting_BPM": rest_b, "Session_Duration_min": dur,
                             "Fat_Percentage": fat, "Water_Intake_liters": water, "BMI": round(bmi, 2)}])[self.feature_cols]

        # 13 columns, pca position and cluster of this person
        enhanced, Xp, Xs = enhance_features(row, self.scaler, self.pca, self.kmeans)
        cluster_id = int(enhanced["cluster_id"].iloc[0])
        calories = float(self.regressor.predict(enhanced)[0])

        # heart/hydration risk only if the user really entered those values
        physio_reliable = not any(k in imputed for k in PHYSIO_KEYS)
        physio_risk = round(float(self.classifier.predict_proba(enhanced)[0, 1]) * 100, 1) if physio_reliable else None
        rule_flags = physio_rule_flags(age, max_b, avg_b, rest_b, water, dur) if physio_reliable else []
        bio_anomaly = bool(physio_reliable and self.iso_forest.predict(Xs)[0] == -1)

        # movement part (from the camera)
        lean = float(data.get("torso_lean_deg", 0.0))
        sym = float(data.get("symmetry_score", 100.0))
        bad = int(data.get("bad_reps", 0))
        mv_risk, mv_flags = movement_risk(lean, sym, bad)
        # overall risk = the bigger of the two
        overall = round(max(physio_risk or 0.0, mv_risk), 1)
        is_anom = bio_anomaly or movement_anomaly(lean, sym, bad)

        return {
            "pca_coordinates": [round(float(v), 3) for v in Xp[0, :3]],
            "cluster_id": cluster_id,
            "cluster_profile": self.cluster_label(cluster_id),
            "predicted_calories_kcal": round(calories, 1),
            "physio_risk_pct": physio_risk,
            "physio_rule_flags": rule_flags,
            "movement_risk_pct": mv_risk,
            "movement_quality_score": round(100.0 - mv_risk, 1),
            "movement_flags": mv_flags,
            "imputed_fields": imputed,
            "injury_risk_probability": overall,
            "injury_risk_label": "High Injury / Overtraining Risk" if overall >= 50 else "Safe & Optimal Performance",
            "is_anomaly": bool(is_anom),
            "anomaly_status": "⚠️ DANGER: Biometric / Pose Anomaly Flagged!" if is_anom else "✅ Normal Pattern",
        }
