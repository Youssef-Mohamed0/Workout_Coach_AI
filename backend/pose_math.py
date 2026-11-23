# limits picked by hand
LEAN_LIMIT_DEG = 17.5
SYMMETRY_FLOOR = 82.0
LEAN_ANOMALY_DEG = 20.0
SYMMETRY_ANOMALY = 76.0
BAD_REPS_ANOMALY = 2


# movement risk from 0 to 100, plus the reasons as text
def movement_risk(torso_lean, symmetry, bad_reps):
    risk, flags = 0.0, []
    if torso_lean > LEAN_LIMIT_DEG:
        risk += (torso_lean - LEAN_LIMIT_DEG) * 2.8
        flags.append(f"torso lean {torso_lean:.1f} deg > {LEAN_LIMIT_DEG}")
    if symmetry < SYMMETRY_FLOOR:
        risk += (SYMMETRY_FLOOR - symmetry) * 1.9
        flags.append(f"left/right symmetry {symmetry:.1f}% < {SYMMETRY_FLOOR}")
    if bad_reps > 0:
        risk += min(45.0, bad_reps * 12.0)
        flags.append(f"{bad_reps} bad rep(s)")
    return round(min(100.0, risk), 1), flags


# True if the movement looks unusual (big lean, low symmetry or 2+ bad reps)
def movement_anomaly(torso_lean, symmetry, bad_reps):
    return bool(torso_lean > LEAN_ANOMALY_DEG or symmetry < SYMMETRY_ANOMALY
                or bad_reps >= BAD_REPS_ANOMALY)
