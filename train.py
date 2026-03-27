"""
F1 Race Predictor — Model Training Script
Builds features from historical F1 data (2021–2025) and trains a podium classifier.
"""

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, fbeta_score, make_scorer, confusion_matrix
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_DIR = "F1 2021-2025 dataset"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def lap_time_to_seconds(t):
    """Convert '1:18.231' → 78.231. Returns NaN if not parseable."""
    if pd.isna(t) or t == "" or t == "\\N":
        return np.nan
    t = str(t).strip()
    m = re.match(r"^(\d+):(\d+\.\d+)$", t)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    try:
        return float(t)
    except ValueError:
        return np.nan


def rolling_dnf_rate(series, window=10):
    """Given a boolean series (1=DNF), compute rolling DNF rate."""
    return series.rolling(window, min_periods=1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: LOAD & MERGE DATA
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1 — Loading data")
print("=" * 60)

races        = pd.read_csv(f"{DATA_DIR}/races.csv", parse_dates=["date"])
results      = pd.read_csv(f"{DATA_DIR}/results.csv")
drivers      = pd.read_csv(f"{DATA_DIR}/drivers.csv")
constructors = pd.read_csv(f"{DATA_DIR}/constructors.csv")
qualifying   = pd.read_csv(f"{DATA_DIR}/qualifying.csv")
circuits     = pd.read_csv(f"{DATA_DIR}/circuits.csv")
status       = pd.read_csv(f"{DATA_DIR}/status.csv")

# Ensure fullName exists in drivers
if "fullName" not in drivers.columns:
    drivers["fullName"] = drivers["forename"].str.strip() + " " + drivers["surname"].str.strip()

print(f"Races: {len(races)} | Results: {len(results)} | Drivers: {len(drivers)}")
print(f"Circuits: {len(circuits)} | Qualifying: {len(qualifying)}")

# ─────────────────────────────────────────────────────────────────────────────
# Build qualifying gap to pole
# ─────────────────────────────────────────────────────────────────────────────

print("\nParsing qualifying times...")

def best_q_time(row):
    """Best qualifying time a driver set (q3 > q2 > q1 priority for actual lap)."""
    for col in ["q3", "q2", "q1"]:
        t = lap_time_to_seconds(row[col])
        if not np.isnan(t):
            return t
    return np.nan

qualifying["best_q_time"] = qualifying.apply(best_q_time, axis=1)

# Pole time = fastest Q time per race (position 1 in qualifying)
pole_times = (
    qualifying[qualifying["position"] == 1][["raceId", "best_q_time"]]
    .rename(columns={"best_q_time": "pole_time"})
)

qualifying = qualifying.merge(pole_times, on="raceId", how="left")
qualifying["quali_gap_to_pole"] = qualifying["best_q_time"] - qualifying["pole_time"]
qualifying["quali_gap_to_pole"] = qualifying["quali_gap_to_pole"].fillna(0.0).clip(lower=0.0)

# Also bring in qualifying position = actual starting grid position
qualifying_slim = qualifying[["raceId", "driverId", "quali_gap_to_pole", "position"]].copy()
qualifying_slim = qualifying_slim.rename(columns={"position": "quali_position"})

# ─────────────────────────────────────────────────────────────────────────────
# Merge main dataframe
# ─────────────────────────────────────────────────────────────────────────────

print("Merging tables...")

df = results.merge(races[["raceId", "year", "round", "circuitId", "date", "name"]], on="raceId", how="left")
df = df.merge(drivers[["driverId", "fullName"]], on="driverId", how="left")
df = df.merge(constructors[["constructorId", "name"]], on="constructorId", how="left", suffixes=("_race", "_team"))
df = df.merge(qualifying_slim, on=["raceId", "driverId"], how="left")
df = df.merge(circuits[["circuitId", "name"]], on="circuitId", how="left", suffixes=("", "_circuit"))

# Fill missing quali gap
df["quali_gap_to_pole"] = df["quali_gap_to_pole"].fillna(0.0)

# Use qualifying position as grid_position (actual starting grid)
# Fall back to median (10) if qualifying data not available
df["grid_position"] = df["quali_position"].fillna(10.0)

# Clean up column names
df = df.rename(columns={
    "name_race": "race_name",
    "name_team": "team_name",
    "name":      "circuit_name",
    "fullName":  "driver_name",
})

# If circuit_name came from circuits merge, handle it
if "name" in df.columns:
    df = df.rename(columns={"name": "circuit_name_x"})

# Sort chronologically — critical to avoid leakage
df = df.sort_values(["date", "raceId", "positionOrder"]).reset_index(drop=True)

print(f"Merged dataframe shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: DEFINE TARGET & REMOVE LEAKAGE
# ─────────────────────────────────────────────────────────────────────────────

# Target: did the driver finish on the podium?
df["podium"] = (df["positionOrder"] <= 3).astype(int)

# DNF flag: statusId not in [1..10] (Finished / lapped)
finished_status_ids = set(range(1, 11))  # 1=Finished, 2-10 = lapped
df["is_dnf"] = (~df["statusId"].isin(finished_status_ids)).astype(int)

print(f"\nPodium rate: {df['podium'].mean():.2%}")
print(f"DNF rate: {df['is_dnf'].mean():.2%}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 3: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2 — Feature engineering")
print("=" * 60)

# Sort by driver + date for rolling calculations
df = df.sort_values(["driverId", "date", "raceId"]).reset_index(drop=True)

# --- Rolling driver stats (computed from history only via shift) ---

# Points per race (actual race points for rolling avg)
df["race_points"] = df["points"]

# Finish position (use positionOrder which is always filled)
df["finish_pos"] = df["positionOrder"]

# Podium flag for rolling
df["podium_flag"] = df["podium"]
df["win_flag"]    = (df["positionOrder"] == 1).astype(int)

# Per-driver rolling features (shift(1) ensures we only use past data)
for driver_id, grp in df.groupby("driverId"):
    idx = grp.index

    df.loc[idx, "rolling_points_5"] = (
        grp["race_points"].shift(1).rolling(5, min_periods=1).mean()
    )
    df.loc[idx, "rolling_finish_5"] = (
        grp["finish_pos"].shift(1).rolling(5, min_periods=1).mean()
    )
    df.loc[idx, "dnf_rate"] = (
        grp["is_dnf"].shift(1).rolling(10, min_periods=1).mean()
    )
    df.loc[idx, "podiums_last_5"] = (
        grp["podium_flag"].shift(1).rolling(5, min_periods=1).sum()
    )
    df.loc[idx, "driver_experience"] = (
        grp["raceId"].shift(1).expanding().count()
    )
    # Career rates (cumulative history only)
    df.loc[idx, "career_podium_rate"] = (
        grp["podium_flag"].shift(1).expanding().mean()
    ).fillna(0.0)
    df.loc[idx, "career_win_rate"] = (
        grp["win_flag"].shift(1).expanding().mean()
    ).fillna(0.0)
    # Average qualifying position last 5 races
    df.loc[idx, "avg_quali_pos_5"] = (
        grp["grid_position"].shift(1).rolling(5, min_periods=1).mean()
    )

# --- Best finish at circuit ---
def best_circuit_finish(grp):
    """Compute best historical finish at this circuit before current race."""
    result = pd.Series(index=grp.index, dtype=float)
    for i, (idx, row) in enumerate(grp.iterrows()):
        past = grp.iloc[:i]
        if len(past) == 0:
            result[idx] = 20.0  # first time at circuit → use 20
        else:
            result[idx] = past["finish_pos"].min()
    return result

print("Computing best circuit finishes per driver (may take a moment)...")
df = df.sort_values(["driverId", "circuitId", "date"]).reset_index(drop=True)
best_circuit = df.groupby(["driverId", "circuitId"]).apply(best_circuit_finish)
# Flatten multi-index
best_circuit = best_circuit.reset_index(level=[0, 1], drop=True)
df["best_finish_at_circuit"] = best_circuit
df["best_finish_at_circuit"] = df["best_finish_at_circuit"].fillna(20.0)

# Re-sort chronologically
df = df.sort_values(["date", "raceId", "positionOrder"]).reset_index(drop=True)

# --- Constructor reliability (DNF rate last 10 races) ---
df = df.sort_values(["constructorId", "date", "raceId"]).reset_index(drop=True)
for ctor_id, grp in df.groupby("constructorId"):
    idx = grp.index
    df.loc[idx, "constructor_reliability"] = (
        grp["is_dnf"].shift(1).rolling(10, min_periods=1).mean()
    )

# Re-sort
df = df.sort_values(["date", "raceId", "positionOrder"]).reset_index(drop=True)

# --- Championship positions (computed cumulatively from points) ---
# Compute cumulative points per driver up to each race
df = df.sort_values(["driverId", "date", "raceId"]).reset_index(drop=True)

for driver_id, grp in df.groupby("driverId"):
    idx = grp.index
    cum_points = grp["race_points"].shift(1).cumsum().fillna(0)
    df.loc[idx, "cumulative_driver_points"] = cum_points

# Per-race: rank drivers by cumulative points at that point in time
df = df.sort_values(["date", "raceId"]).reset_index(drop=True)

def assign_champ_pos(grp):
    grp = grp.copy()
    grp["driver_championship_pos"] = grp["cumulative_driver_points"].rank(
        ascending=False, method="min"
    ).clip(upper=20)
    return grp

df = df.groupby("raceId", group_keys=False).apply(assign_champ_pos)

# Constructor championship pos
df = df.sort_values(["constructorId", "date", "raceId"]).reset_index(drop=True)
for ctor_id, grp in df.groupby("constructorId"):
    idx = grp.index
    cum_pts = grp["race_points"].shift(1).cumsum().fillna(0)
    df.loc[idx, "cumulative_ctor_points"] = cum_pts

df = df.sort_values(["date", "raceId"]).reset_index(drop=True)

def assign_ctor_pos(grp):
    grp = grp.copy()
    grp["constructor_championship_pos"] = grp["cumulative_ctor_points"].rank(
        ascending=False, method="min"
    ).clip(upper=10)
    return grp

df = df.groupby("raceId", group_keys=False).apply(assign_ctor_pos)

# Re-sort
df = df.sort_values(["date", "raceId", "positionOrder"]).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# ENCODE CATEGORICAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

print("Encoding categorical features...")

label_encoders = {}

for col, col_name in [("driver_name", "driver_encoded"),
                      ("team_name", "constructor_encoded"),
                      ("circuitId", "circuit_id")]:
    le = LabelEncoder()
    df[col_name] = le.fit_transform(df[col].astype(str))
    label_encoders[col_name] = le

# ─────────────────────────────────────────────────────────────────────────────
# FILL REMAINING MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────

feature_cols = [
    "circuit_id",
    "driver_encoded",
    "rolling_points_5",
    "rolling_finish_5",
    "dnf_rate",
    "driver_experience",
    "quali_gap_to_pole",
    "driver_championship_pos",
    "constructor_championship_pos",
    "podiums_last_5",
    "best_finish_at_circuit",
    "career_podium_rate",
    "career_win_rate",
]

for col in feature_cols:
    if col in df.columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

print(f"\nFeature summary:")
for col in feature_cols:
    print(f"  {col:35s} missing={df[col].isna().sum():3d}  mean={df[col].mean():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 4: TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3 — Training models")
print("=" * 60)

X = df[feature_cols].values
y = df["podium"].values

print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Positive class (podium): {y.sum()} / {len(y)} ({y.mean():.2%})")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# TimeSeriesSplit — preserve temporal order
tscv = TimeSeriesSplit(n_splits=5)

pos_weight = (y == 0).sum() / (y == 1).sum()

# ── Tune XGBoost via RandomizedSearchCV (TimeSeriesSplit) ────────────────────
# Use F-beta score with beta=0.5 — weights precision twice as heavily as recall
precision_focused = make_scorer(fbeta_score, beta=0.5, zero_division=0)

print("Tuning XGBoost hyperparameters (RandomizedSearchCV, 50 iterations, precision-focused)...")
xgb_param_dist = {
    "n_estimators":     [200, 300, 400, 500],
    "max_depth":        [3, 4, 5],
    "learning_rate":    [0.02, 0.05, 0.08, 0.10],
    "min_child_weight": [3, 5, 7, 10],
    "gamma":            [0.1, 0.2, 0.3, 0.5],
    "scale_pos_weight": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
    "colsample_bytree": [0.6, 0.7, 0.8],
    "reg_alpha":        [0.05, 0.10, 0.50, 1.0],
    "reg_lambda":       [1.0, 1.5, 2.0, 3.0],
    "subsample":        [0.7, 0.8, 0.9],
}
xgb_search = RandomizedSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1),
    param_distributions=xgb_param_dist,
    n_iter=50,
    scoring=precision_focused,
    cv=TimeSeriesSplit(n_splits=5),
    random_state=42,
    verbose=1,
    n_jobs=-1,
)
xgb_search.fit(X_scaled, y)
print(f"Best XGBoost params: {xgb_search.best_params_}")
best_xgb = xgb_search.best_estimator_

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, subsample=0.8,
        learning_rate=0.05, random_state=42
    ),
    "XGBoost": best_xgb,
}

reports = {}
f1_scores = {}
fold_metrics = {}

for name, model in models.items():
    print(f"\n--- {name} ---")
    splits = list(tscv.split(X_scaled))

    # Per-fold precision, recall, F1
    fold_rows = []
    all_y_true, all_y_pred = [], []
    for fold_num, (train_idx, test_idx) in enumerate(splits, 1):
        model.fit(X_scaled[train_idx], y[train_idx])
        y_pred = model.predict(X_scaled[test_idx])
        all_y_true.extend(y[test_idx])
        all_y_pred.extend(y_pred)
        p = precision_score(y[test_idx], y_pred, zero_division=0)
        r = recall_score(y[test_idx], y_pred, zero_division=0)
        f = f1_score(y[test_idx], y_pred, zero_division=0)
        fold_rows.append((fold_num, p, r, f))
        print(f"  Fold {fold_num}: Precision={p:.3f}  Recall={r:.3f}  F1={f:.3f}")

    # Confusion matrix across all folds combined
    cm = confusion_matrix(all_y_true, all_y_pred)
    tn, fp, fn, tp = cm.ravel()
    fold_metrics[name] = (fold_rows, int(tp), int(fp), int(fn), int(tn))
    mean_f1 = sum(row[3] for row in fold_rows) / len(fold_rows)
    print(f"Mean F1: {mean_f1:.3f}")

    # Classification report on last fold (already fitted above)
    train_idx, test_idx = splits[-1]
    model.fit(X_scaled[train_idx], y[train_idx])
    y_pred = model.predict(X_scaled[test_idx])
    report = classification_report(y[test_idx], y_pred, target_names=["No Podium", "Podium"])
    print(report)
    reports[name] = report
    f1_scores[name] = mean_f1

# Pick best model — prefer Logistic Regression if it ties (gives calibrated probabilities)
best_score = max(f1_scores.values())
tied = [n for n, s in f1_scores.items() if abs(s - best_score) < 0.01]
if "Logistic Regression" in tied:
    best_name = "Logistic Regression"
else:
    best_name = max(f1_scores, key=f1_scores.get)
print(f"\n{'='*60}")
print(f"Best model: {best_name} (F1={f1_scores[best_name]:.3f})")
print(f"{'='*60}")

# Retrain best model on all data
best_model = models[best_name]
best_model.fit(X_scaled, y)

# Feature importance
print("\nTop 10 Most Important Features:")
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
elif hasattr(best_model, "coef_"):
    importances = np.abs(best_model.coef_[0])
else:
    importances = np.ones(len(feature_cols))

feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
for rank, (feat, imp) in enumerate(feat_imp[:10], 1):
    print(f"  {rank:2d}. {feat:35s} {imp:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 5: SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4 — Saving artifacts")
print("=" * 60)

with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("feature_columns.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

# Save model report
report_lines = [
    f"F1 Race Predictor — Model Report",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"Training data: 2021–2025 F1 seasons",
    f"Dataset size: {len(df)} rows, {len(feature_cols)} features",
    f"",
    f"{'='*60}",
    f"MODEL COMPARISON (TimeSeriesSplit CV F1)",
    f"{'='*60}",
]
for name, score in sorted(f1_scores.items(), key=lambda x: x[1], reverse=True):
    marker = " <-- SELECTED" if name == best_name else ""
    report_lines.append(f"  {name:30s}: F1 = {score:.3f}{marker}")

report_lines += [
    f"",
    f"{'='*60}",
    f"BEST MODEL: {best_name}",
    f"{'='*60}",
    f"",
    f"Classification Report (final fold):",
    reports[best_name],
    f"",
    f"{'='*60}",
    f"PER-FOLD METRICS ({best_name})",
    f"{'='*60}",
    f"  {'Fold':>6}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}",
]
best_fold_rows, best_tp, best_fp, best_fn, best_tn = fold_metrics[best_name]
for fold_num, p, r, f in best_fold_rows:
    report_lines.append(f"  {'Fold '+str(fold_num):>6}  {p:>10.3f}  {r:>8.3f}  {f:>8.3f}")

report_lines += [
    f"",
    f"{'='*60}",
    f"CONFUSION MATRIX ({best_name}, all folds combined)",
    f"{'='*60}",
    f"  TP={best_tp}  FP={best_fp}  FN={best_fn}  TN={best_tn}",
    f"",
    f"{'='*60}",
    f"FEATURE IMPORTANCES",
    f"{'='*60}",
]
for rank, (feat, imp) in enumerate(feat_imp, 1):
    report_lines.append(f"  {rank:2d}. {feat:35s} {imp:.4f}")

with open("model_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

# Also save unique driver/circuit lists for the app
driver_list = sorted(df["driver_name"].dropna().unique().tolist())

# Build circuit list using friendly RACE names (e.g. "Bahrain Grand Prix") not circuit names
# Map circuitId -> most recent race name
latest_race_name = (
    races.sort_values("year")
         .groupby("circuitId")["name"]
         .last()
         .reset_index()
         .rename(columns={"name": "race_name"})
)
circuit_list_df = (
    df[["circuitId", "circuit_name"]]
    .drop_duplicates()
    .merge(latest_race_name, on="circuitId", how="left")
)
circuit_list_df["display_name"] = circuit_list_df["race_name"].fillna(circuit_list_df["circuit_name"])
circuit_list = sorted(
    circuit_list_df[["circuitId", "display_name"]].values.tolist(),
    key=lambda x: x[1],
)

with open("driver_list.pkl", "wb") as f:
    pickle.dump(driver_list, f)

with open("circuit_list.pkl", "wb") as f:
    pickle.dump(circuit_list, f)

# Save driver stats (for Tab 2 — full grid prediction)
driver_stats = df.sort_values("date").groupby("driver_name").last()[[
    "rolling_points_5", "rolling_finish_5", "dnf_rate",
    "constructor_reliability", "driver_experience",
    "podiums_last_5", "best_finish_at_circuit",
    "quali_gap_to_pole", "driver_championship_pos",
    "constructor_championship_pos", "driver_encoded",
    "constructor_encoded", "team_name",
    "career_podium_rate", "career_win_rate", "avg_quali_pos_5",
]]
driver_stats.to_csv("driver_stats.csv")

# Circuit → circuitId mapping (indexed by display name = race name)
circuit_map = circuit_list_df.set_index("display_name")[["circuitId"]]
circuit_map.to_csv("circuit_map.csv")

print("Saved: model.pkl, scaler.pkl, label_encoders.pkl, feature_columns.pkl")
print("Saved: model_report.txt, driver_list.pkl, circuit_list.pkl")
print("Saved: driver_stats.csv, circuit_map.csv")

# ─────────────────────────────────────────────────────────────────────────────
# PART 6: TEST PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5 — Test predictions")
print("=" * 60)

def predict_podium(driver_name, circuit_name, grid_position,
                   quali_gap=0.0, champ_pos=10, ctor_pos=5,
                   podiums_last5=0, best_circuit_finish=20):
    """Make a single prediction."""
    # Encode driver
    le_driver = label_encoders["driver_encoded"]
    le_ctor   = label_encoders["constructor_encoded"]
    le_circuit = label_encoders["circuit_id"]

    if driver_name not in le_driver.classes_:
        return None, None

    stats = driver_stats.loc[driver_name] if driver_name in driver_stats.index else None

    driver_enc = le_driver.transform([driver_name])[0]

    # Get circuitId from name (convert to str since LabelEncoder was trained on str)
    cmap = circuit_map
    if circuit_name in cmap.index:
        cid = str(int(cmap.loc[circuit_name, "circuitId"]))
        if cid in le_circuit.classes_:
            circuit_enc = le_circuit.transform([cid])[0]
        else:
            circuit_enc = 0
    else:
        circuit_enc = 0

    ctor_enc = int(stats["constructor_encoded"]) if stats is not None else 0

    row = {
        "circuit_id":                  circuit_enc,
        "driver_encoded":              driver_enc,
        "constructor_encoded":         ctor_enc,
        "rolling_points_5":            float(stats["rolling_points_5"]) if stats is not None else 5.0,
        "rolling_finish_5":            float(stats["rolling_finish_5"]) if stats is not None else 10.0,
        "dnf_rate":                    float(stats["dnf_rate"]) if stats is not None else 0.05,
        "constructor_reliability":     float(stats["constructor_reliability"]) if stats is not None else 0.05,
        "driver_experience":           float(stats["driver_experience"]) if stats is not None else 50.0,
        "quali_gap_to_pole":           quali_gap,
        "driver_championship_pos":     champ_pos,
        "constructor_championship_pos": ctor_pos,
        "podiums_last_5":              podiums_last5,
        "best_finish_at_circuit":      best_circuit_finish,
        "career_podium_rate":          float(stats["career_podium_rate"]) if stats is not None else 0.0,
        "career_win_rate":             float(stats["career_win_rate"]) if stats is not None else 0.0,
        "avg_quali_pos_5":             float(stats["avg_quali_pos_5"]) if stats is not None else 10.0,
    }

    X_input = np.array([[row[c] for c in feature_cols]])
    X_input_scaled = scaler.transform(X_input)
    prob = best_model.predict_proba(X_input_scaled)[0][1]
    pred = int(prob >= 0.5)
    return pred, prob

# Test 3 predictions
tests = [
    ("Max Verstappen",  "Bahrain Grand Prix", 1, 0.0, 1, 1, 2, 1),
    ("Lewis Hamilton",  "Monaco Grand Prix",  3, 0.3, 2, 1, 1, 2),
    ("Lance Stroll",    "British Grand Prix", 15, 1.2, 12, 7, 0, 18),
]

for driver, circuit, grid, qgap, cp, ctorp, podlast5, bestcirc in tests:
    pred, prob = predict_podium(driver, circuit, grid, qgap, cp, ctorp, podlast5, bestcirc)
    if pred is None:
        print(f"  {driver} at {circuit}: driver not found in training data")
    else:
        result = "PODIUM" if pred == 1 else "No podium"
        print(f"  {driver:20s} at {circuit:25s} grid={grid:2d} => {result} ({prob:.1%} confidence)")

print("\nTraining complete. All artifacts saved.")
