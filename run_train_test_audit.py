"""
run_train_test_audit.py — Complete train/test metric audit for GRACE manuscript.

Runs the full pipeline (cleaning → rating features → model training) and reports
BOTH training and testing metrics for all models, with early stopping to reduce
overfitting and ensure valid generalization estimates.

Usage:
    python run_train_test_audit.py
"""

import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# Rating hyperparameters (same as manuscript)
K0, V_GAIN, RD0, RD_MIN, C_DECAY, K_ITEM = 0.35, 0.04, 0.35, 0.06, 0.010, 0.02
W_THETA_D, W_THETA_S = 0.5, 0.5
W_DIFF_SKILL, W_DIFF_TTYPE = 0.7, 0.3
W_UPDATE_GLOBAL, W_UPDATE_DISC, W_UPDATE_SKILL = 0.5, 1.0, 1.5
W_DIFF_UPDATE, W_DIFF_TT_UPDATE = 1.0, 0.5

PSEUDO_SKILL_NAMES = [
    'Номер теста', 'Длительность', 'Темы', 'Темы среза/дз (прошлый урок)',
    'Количественные характеристики', 'Ментор', 'Туториал', 'Время потока',
    'Оценка учителя', 'Школа', 'Отдел Менторов', 'Обратная связь получена',
    'КРА', 'МКЕ', 'Оценка среза', 'Баллы за тест', 'Попытки', 'Ранг',
    'Комментарий', 'Комментарий.', 'Срез Общее',
]

RATING_FEATURES = [
    "theta_g", "theta_disc", "theta_skill", "rd_g", "rd_disc",
    "diff_skill", "diff_ttype", "e_hat", "surprise5", "days_gap",
    "n_global", "n_disc", "n_skill", "month", "dayofweek",
    "disc_code", "skill_code", "ttype_code",
]

AGG_FEATURES = [
    "roll3", "exp_avg", "lag1", "month", "dayofweek", "test_num",
    "disc_code", "skill_code", "ttype_code",
]

GRACE_FEATURES = RATING_FEATURES + [
    "roll3", "exp_avg", "lag1", "test_num",
    "theta_range", "score_momentum", "gap_bucket", "is_cold",
]

OUT = Path("results")
OUT.mkdir(exist_ok=True)


# ======================================================================
# STEP 1: DATA LOADING & CLEANING
# ======================================================================
def load_and_clean():
    print("=" * 70)
    print("STEP 1: DATA LOADING & CLEANING")
    print("=" * 70)

    df = pd.read_csv("report/student_performance_data.csv", low_memory=False)
    raw = len(df)
    print(f"  Raw export: {raw:,} rows")

    df = df[~df["skill_name"].isin(PSEUDO_SKILL_NAMES)].copy()
    print(f"  After pseudo-skill removal: {len(df):,}")

    df["test_datetime"] = pd.to_datetime(df["test_datetime"], errors="coerce")
    df = df.dropna(subset=["test_datetime"])
    df = df[df["test_datetime"] >= "2018-01-01"]

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["max_score"] = pd.to_numeric(df["max_score"], errors="coerce")
    df = df.dropna(subset=["score", "max_score"])
    df = df[(df["max_score"] > 0) & (df["score"] >= 0) & (df["score"] <= df["max_score"])]
    df = df.drop_duplicates()
    df["pct"] = 100.0 * df["score"] / df["max_score"]

    print(f"  Clean corpus: {len(df):,} rows, {df['student_id'].nunique():,} students")
    print(f"  Date range: {df['test_datetime'].min().date()} to {df['test_datetime'].max().date()}")
    print(f"  Score: mean={df['pct'].mean():.1f}%, std={df['pct'].std():.1f}%")
    return df


# ======================================================================
# STEP 2: RATING FEATURES (pure Python — reliable)
# ======================================================================
def build_rating_features(df):
    print("\n" + "=" * 70)
    print("STEP 2: RATING FEATURE ENGINEERING")
    print("=" * 70)
    t0 = time.time()

    df = df.sort_values("test_datetime", kind="mergesort")
    rid = df.index.to_numpy()
    df = df.reset_index(drop=True)
    y = (df["score"] / df["max_score"]).clip(0, 1).to_numpy()
    ts = pd.to_datetime(df["test_datetime"]).astype("int64").to_numpy() / 86.4e12

    students = df["student_id"].to_numpy()
    discs = df["discipline"].astype(str).to_numpy()
    skills = df["skill_name"].astype(str).to_numpy()
    ttypes = df["test_type_category_id"].astype(str).to_numpy()

    theta_g, theta_d, theta_s = {}, {}, {}
    rd_g_v, rd_d_v = {}, {}
    diff_skill, diff_ttype = {}, {}
    last_seen = {}
    n_g, n_d, n_s = {}, {}, {}
    surprise = {}

    n = len(df)
    out = np.zeros((n, 13), dtype=np.float64)

    for t in range(n):
        if t % 500000 == 0 and t > 0:
            print(f"    {t:,}/{n:,} ({100*t/n:.0f}%) [{time.time()-t0:.0f}s]")

        i = students[t]
        s_key = (i, discs[t])
        sk = (i, skills[t])
        j = skills[t]
        c = ttypes[t]

        th_g = theta_g.get(i, 0.0)
        th_d = theta_d.get(s_key, th_g)
        th_s = theta_s.get(sk, th_d)
        dj = diff_skill.get(j, 0.0)
        dc = diff_ttype.get(c, 0.0)

        gap = ts[t] - last_seen.get(i, ts[t])
        v_g = min(rd_g_v.get(i, RD0**2) + C_DECAY * gap, RD0**2)
        v_d = min(rd_d_v.get(s_key, RD0**2) + C_DECAY * gap, RD0**2)

        e_hat = 1.0 / (1.0 + np.exp(-(W_THETA_D * th_d + W_THETA_S * th_s
                                       - W_DIFF_SKILL * dj - W_DIFF_TTYPE * dc)))
        sur = surprise.get(s_key, [])

        out[t] = [th_g, th_d, th_s,
                  np.sqrt(v_g), np.sqrt(v_d),
                  dj, dc, e_hat,
                  np.mean(sur[-5:]) if sur else 0.0,
                  min(gap, 365.0),
                  n_g.get(i, 0), n_d.get(s_key, 0), n_s.get(sk, 0)]

        err = y[t] - e_hat
        k_g_val = K0 * v_g / (v_g + V_GAIN)
        k_d_val = K0 * v_d / (v_d + V_GAIN)
        theta_g[i] = th_g + W_UPDATE_GLOBAL * k_g_val * err
        theta_d[s_key] = th_d + W_UPDATE_DISC * k_d_val * err
        theta_s[sk] = th_s + W_UPDATE_SKILL * k_d_val * err
        diff_skill[j] = dj - W_DIFF_UPDATE * K_ITEM * err
        diff_ttype[c] = dc - W_DIFF_TT_UPDATE * K_ITEM * err
        rd_g_v[i] = max(v_g * V_GAIN / (v_g + V_GAIN), RD_MIN**2)
        rd_d_v[s_key] = max(v_d * V_GAIN / (v_d + V_GAIN), RD_MIN**2)
        last_seen[i] = ts[t]
        n_g[i] = n_g.get(i, 0) + 1
        n_d[s_key] = n_d.get(s_key, 0) + 1
        n_s[sk] = n_s.get(sk, 0) + 1
        surprise[s_key] = (sur + [err])[-5:]

    elapsed = time.time() - t0
    print(f"    Rating loop: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    cols = ["theta_g", "theta_disc", "theta_skill", "rd_g", "rd_disc",
            "diff_skill", "diff_ttype", "e_hat", "surprise5",
            "days_gap", "n_global", "n_disc", "n_skill"]
    feats = pd.DataFrame(out, columns=cols)

    dt = pd.to_datetime(df["test_datetime"])
    feats["month"] = dt.dt.month.values
    feats["dayofweek"] = dt.dt.dayofweek.values
    for src, col in [("discipline", "disc_code"), ("skill_name", "skill_code"),
                     ("test_type_category_id", "ttype_code")]:
        feats[col] = df[src].astype("category").cat.codes.values
    feats["y"] = y * 100.0
    feats["ts"] = dt.values
    feats["student_id"] = df["student_id"].values
    feats["discipline"] = df["discipline"].values
    feats["rid"] = rid

    print(f"  Rating features: {len(feats):,} rows x {len(cols)} rating columns")
    return feats


# ======================================================================
# STEP 3: AGGREGATE FEATURES
# ======================================================================
def build_aggregate_features(df):
    df = df.sort_values(["student_id", "discipline", "test_datetime"]).reset_index()
    y = (df["score"] / df["max_score"]).clip(0, 1) * 100
    df["_y"] = y
    g = df.groupby(["student_id", "discipline"])["_y"]

    df["roll3"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["exp_avg"] = g.transform(lambda s: s.shift(1).expanding().mean())
    df["lag1"] = g.transform(lambda s: s.shift(1))

    dt = pd.to_datetime(df["test_datetime"])
    df["month"] = dt.dt.month
    df["dayofweek"] = dt.dt.dayofweek
    df["test_num"] = df.groupby(["student_id", "discipline"]).cumcount()

    for src, dst in [("discipline", "disc_code"), ("skill_name", "skill_code"),
                     ("test_type_category_id", "ttype_code")]:
        df[dst] = df[src].astype("category").cat.codes

    out = df[["roll3", "exp_avg", "lag1", "month", "dayofweek", "test_num",
              "disc_code", "skill_code", "ttype_code", "_y", "index",
              "student_id", "discipline"]].copy()
    out = out.rename(columns={"_y": "y", "index": "rid"})
    out["ts"] = dt.values
    return out.sort_values("ts").reset_index(drop=True)


def build_combined(feats, agg):
    combined = feats.merge(
        agg[["rid", "roll3", "exp_avg", "lag1", "test_num"]],
        on="rid", how="left"
    )
    combined["theta_range"] = combined["theta_g"] - combined["theta_disc"]
    combined["score_momentum"] = combined["surprise5"] * combined["n_disc"].clip(upper=20)
    combined["gap_bucket"] = pd.cut(combined["days_gap"],
                                     bins=[-1, 0, 1, 3, 7, 14, 30, 365],
                                     labels=False).fillna(6).astype(int)
    combined["is_cold"] = (combined["n_disc"] < 3).astype(int)
    return combined


# ======================================================================
# EVALUATION HELPERS
# ======================================================================
def chrono_split(data, train_frac=0.7, cal_frac=0.1):
    n = len(data)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + cal_frac))
    return data.iloc[:i1].copy(), data.iloc[i1:i2].copy(), data.iloc[i2:].copy()


def eval_metrics(y_true, y_pred):
    return r2_score(y_true, y_pred), mean_absolute_error(y_true, y_pred)


def print_row(model, split, r2, mae, n, gap_r2=None, gap_mae=None):
    gap_str = ""
    if gap_r2 is not None:
        gap_str = f"  (gap: R²={gap_r2:+.3f}, MAE={gap_mae:+.2f})"
    print(f"  {model:<30s} {split:<6s}  R²={r2:.4f}  MAE={mae:.2f}pp  n={n:>10,}{gap_str}")


# ======================================================================
# MAIN
# ======================================================================
def main():
    t_start = time.time()

    # --- Load & clean ---
    df = load_and_clean()

    # --- Build features ---
    rating_feats = build_rating_features(df)
    agg_feats = build_aggregate_features(df)
    grace_feats = build_combined(rating_feats, agg_feats)

    print(f"\n  GRACE combined features: {len(grace_feats):,} rows x {len(GRACE_FEATURES)} features")

    # --- Chronological splits ---
    # GRACE: 70/10/20
    tr_g, cal_g, te_g = chrono_split(grace_feats, 0.7, 0.1)
    # Baseline: 70/10/20 on aggregate-only (drop NaN first-attempts)
    agg_clean = agg_feats.dropna(subset=["exp_avg"])
    tr_a, _, te_a = chrono_split(agg_clean, 0.7, 0.1)

    print(f"\n  GRACE split:    train={len(tr_g):,}  cal={len(cal_g):,}  test={len(te_g):,}")
    print(f"  Baseline split: train={len(tr_a):,}  test={len(te_a):,}")
    print(f"  Train period: {tr_g['ts'].min().date()} to {tr_g['ts'].max().date()}")
    print(f"  Test period:  {te_g['ts'].min().date()} to {te_g['ts'].max().date()}")

    # ==================================================================
    # MODEL TRAINING WITH EARLY STOPPING
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 3: MODEL TRAINING (with early stopping on calibration set)")
    print("=" * 70)

    results = {}

    # --- 1. XGBoost Baseline (with early stopping) ---
    print("\n  Training XGBoost Baseline...")
    xgb_base = XGBRegressor(
        n_estimators=2000, learning_rate=0.04, max_depth=10,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        reg_lambda=1.5, reg_alpha=0.1, objective="reg:squarederror",
        tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0,
        early_stopping_rounds=50,
    )
    # Use cal set for early stopping validation
    cal_a = agg_clean.iloc[int(len(agg_clean)*0.7):int(len(agg_clean)*0.8)]
    xgb_base.fit(
        tr_a[AGG_FEATURES].fillna(-999), tr_a["y"],
        eval_set=[(cal_a[AGG_FEATURES].fillna(-999), cal_a["y"])],
        verbose=False,
    )
    best_trees_base = xgb_base.best_iteration
    print(f"    Best iteration: {best_trees_base} / 2000")

    pred_tr_base = xgb_base.predict(tr_a[AGG_FEATURES].fillna(-999))
    pred_te_base = xgb_base.predict(te_a[AGG_FEATURES].fillna(-999))
    r2_tr, mae_tr = eval_metrics(tr_a["y"], pred_tr_base)
    r2_te, mae_te = eval_metrics(te_a["y"], pred_te_base)
    results["XGB Baseline"] = {"train": (r2_tr, mae_tr, len(tr_a)),
                                "test": (r2_te, mae_te, len(te_a))}

    # --- 2. XGBoost Baseline WITHOUT early stopping (original manuscript) ---
    print("\n  Training XGBoost Baseline (no early stopping, manuscript config)...")
    xgb_base_orig = XGBRegressor(
        n_estimators=1000, learning_rate=0.04, max_depth=10,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        reg_lambda=1.5, reg_alpha=0.1, objective="reg:squarederror",
        tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0,
    )
    xgb_base_orig.fit(tr_a[AGG_FEATURES].fillna(-999), tr_a["y"])

    pred_tr_orig = xgb_base_orig.predict(tr_a[AGG_FEATURES].fillna(-999))
    pred_te_orig = xgb_base_orig.predict(te_a[AGG_FEATURES].fillna(-999))
    r2_tr_o, mae_tr_o = eval_metrics(tr_a["y"], pred_tr_orig)
    r2_te_o, mae_te_o = eval_metrics(te_a["y"], pred_te_orig)
    results["XGB Base (no ES)"] = {"train": (r2_tr_o, mae_tr_o, len(tr_a)),
                                    "test": (r2_te_o, mae_te_o, len(te_a))}

    # --- 3. Random Forest ---
    print("\n  Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=20, min_samples_leaf=25,
        max_features="sqrt", n_jobs=-1, random_state=SEED,
    )
    RF_SUB = 300_000
    tr_rf = tr_a.sample(min(RF_SUB, len(tr_a)), random_state=SEED)
    rf.fit(tr_rf[AGG_FEATURES].fillna(-999), tr_rf["y"])

    pred_tr_rf = rf.predict(tr_a[AGG_FEATURES].fillna(-999))
    pred_te_rf = rf.predict(te_a[AGG_FEATURES].fillna(-999))
    r2_tr_r, mae_tr_r = eval_metrics(tr_a["y"], pred_tr_rf)
    r2_te_r, mae_te_r = eval_metrics(te_a["y"], pred_te_rf)
    results["Random Forest"] = {"train": (r2_tr_r, mae_tr_r, len(tr_a)),
                                 "test": (r2_te_r, mae_te_r, len(te_a))}

    # --- 4. GRACE XGBoost (with early stopping) ---
    print("\n  Training GRACE (XGBoost + rating features, with early stopping)...")
    xgb_grace = XGBRegressor(
        n_estimators=2000, learning_rate=0.04, max_depth=10,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        reg_lambda=1.5, reg_alpha=0.1, objective="reg:squarederror",
        tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0,
        early_stopping_rounds=50,
    )
    xgb_grace.fit(
        tr_g[GRACE_FEATURES].fillna(-999), tr_g["y"],
        eval_set=[(cal_g[GRACE_FEATURES].fillna(-999), cal_g["y"])],
        verbose=False,
    )
    best_trees_grace = xgb_grace.best_iteration
    print(f"    Best iteration: {best_trees_grace} / 2000")

    pred_tr_grace = xgb_grace.predict(tr_g[GRACE_FEATURES].fillna(-999))
    pred_te_grace = xgb_grace.predict(te_g[GRACE_FEATURES].fillna(-999))
    r2_tr_g, mae_tr_g = eval_metrics(tr_g["y"], pred_tr_grace)
    r2_te_g, mae_te_g = eval_metrics(te_g["y"], pred_te_grace)
    results["GRACE (early stop)"] = {"train": (r2_tr_g, mae_tr_g, len(tr_g)),
                                      "test": (r2_te_g, mae_te_g, len(te_g))}

    # --- 5. GRACE XGBoost (original manuscript config, no early stopping) ---
    print("\n  Training GRACE (manuscript config, no early stopping)...")
    xgb_grace_orig = XGBRegressor(
        n_estimators=1000, learning_rate=0.04, max_depth=10,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        reg_lambda=1.5, reg_alpha=0.1, objective="reg:squarederror",
        tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0,
    )
    xgb_grace_orig.fit(tr_g[GRACE_FEATURES].fillna(-999), tr_g["y"])

    pred_tr_go = xgb_grace_orig.predict(tr_g[GRACE_FEATURES].fillna(-999))
    pred_te_go = xgb_grace_orig.predict(te_g[GRACE_FEATURES].fillna(-999))
    r2_tr_go, mae_tr_go = eval_metrics(tr_g["y"], pred_tr_go)
    r2_te_go, mae_te_go = eval_metrics(te_g["y"], pred_te_go)
    results["GRACE (no ES)"] = {"train": (r2_tr_go, mae_tr_go, len(tr_g)),
                                 "test": (r2_te_go, mae_te_go, len(te_g))}

    # ==================================================================
    # RESULTS TABLE
    # ==================================================================
    print("\n" + "=" * 70)
    print("TRAIN vs TEST ACCURACY — COMPLETE AUDIT")
    print("=" * 70)

    print(f"\n  {'Model':<30s} {'Split':<6s} {'R²':>8s} {'MAE(pp)':>8s} {'n':>12s} {'R² gap':>8s} {'MAE gap':>8s}")
    print("  " + "-" * 78)

    for name, vals in results.items():
        tr_r2, tr_mae, tr_n = vals["train"]
        te_r2, te_mae, te_n = vals["test"]
        gap_r2 = tr_r2 - te_r2
        gap_mae = te_mae - tr_mae
        print(f"  {name:<30s} {'TRAIN':<6s} {tr_r2:>8.4f} {tr_mae:>8.2f} {tr_n:>12,}")
        print(f"  {'':<30s} {'TEST':<6s} {te_r2:>8.4f} {te_mae:>8.2f} {te_n:>12,}")
        print(f"  {'':<30s} {'GAP':<6s} {gap_r2:>+8.4f} {gap_mae:>+8.2f}")
        print()

    # ==================================================================
    # ANALYSIS: OVERFITTING DIAGNOSIS
    # ==================================================================
    print("=" * 70)
    print("OVERFITTING DIAGNOSIS")
    print("=" * 70)

    print("""
  RULE OF THUMB:
    R2 gap < 0.05  = minimal overfitting (excellent generalization)
    R2 gap 0.05-0.15 = moderate overfitting (acceptable with large data)
    R2 gap > 0.15  = significant overfitting (needs regularization)
    """)

    for name, vals in results.items():
        tr_r2, _, _ = vals["train"]
        te_r2, _, _ = vals["test"]
        gap = tr_r2 - te_r2
        if gap < 0.05:
            verdict = "MINIMAL overfitting — excellent generalization"
        elif gap < 0.15:
            verdict = "MODERATE overfitting — acceptable for this data scale"
        else:
            verdict = "SIGNIFICANT overfitting — consider stronger regularization"
        print(f"  {name:<30s}: gap={gap:+.3f} → {verdict}")

    # ==================================================================
    # EARLY STOPPING IMPACT
    # ==================================================================
    print("\n" + "=" * 70)
    print("EARLY STOPPING IMPACT")
    print("=" * 70)

    _, _, tr_n = results["XGB Base (no ES)"]["train"]
    print(f"\n  XGBoost Baseline:")
    print(f"    Without ES: train R²={results['XGB Base (no ES)']['train'][0]:.4f}, test R²={results['XGB Base (no ES)']['test'][0]:.4f}, gap={results['XGB Base (no ES)']['train'][0]-results['XGB Base (no ES)']['test'][0]:+.4f}")
    print(f"    With ES:    train R²={results['XGB Baseline']['train'][0]:.4f}, test R²={results['XGB Baseline']['test'][0]:.4f}, gap={results['XGB Baseline']['train'][0]-results['XGB Baseline']['test'][0]:+.4f}")
    print(f"    ES trees: {best_trees_base} vs fixed 1000")

    print(f"\n  GRACE:")
    print(f"    Without ES: train R²={results['GRACE (no ES)']['train'][0]:.4f}, test R²={results['GRACE (no ES)']['test'][0]:.4f}, gap={results['GRACE (no ES)']['train'][0]-results['GRACE (no ES)']['test'][0]:+.4f}")
    print(f"    With ES:    train R²={results['GRACE (early stop)']['train'][0]:.4f}, test R²={results['GRACE (early stop)']['test'][0]:.4f}, gap={results['GRACE (early stop)']['train'][0]-results['GRACE (early stop)']['test'][0]:+.4f}")
    print(f"    ES trees: {best_trees_grace} vs fixed 1000")

    es_delta_base = results["XGB Baseline"]["test"][0] - results["XGB Base (no ES)"]["test"][0]
    es_delta_grace = results["GRACE (early stop)"]["test"][0] - results["GRACE (no ES)"]["test"][0]
    print(f"\n  Early stopping test R² improvement:")
    print(f"    Baseline: {es_delta_base:+.4f}")
    print(f"    GRACE:    {es_delta_grace:+.4f}")

    # ==================================================================
    # SUMMARY FOR MANUSCRIPT
    # ==================================================================
    print("\n" + "=" * 70)
    print("MANUSCRIPT-READY SUMMARY")
    print("=" * 70)

    best_grace = "GRACE (early stop)" if results["GRACE (early stop)"]["test"][0] > results["GRACE (no ES)"]["test"][0] else "GRACE (no ES)"
    best_base = "XGB Baseline" if results["XGB Baseline"]["test"][0] > results["XGB Base (no ES)"]["test"][0] else "XGB Base (no ES)"

    bg = results[best_grace]
    bb = results[best_base]
    br = results["Random Forest"]

    print(f"\n  Best GRACE config:    {best_grace}")
    print(f"    Train: R²={bg['train'][0]:.3f}, MAE={bg['train'][1]:.2f}")
    print(f"    Test:  R²={bg['test'][0]:.3f}, MAE={bg['test'][1]:.2f}")
    print(f"    Gap:   R²={bg['train'][0]-bg['test'][0]:+.3f}")

    print(f"\n  Best Baseline config: {best_base}")
    print(f"    Train: R²={bb['train'][0]:.3f}, MAE={bb['train'][1]:.2f}")
    print(f"    Test:  R²={bb['test'][0]:.3f}, MAE={bb['test'][1]:.2f}")
    print(f"    Gap:   R²={bb['train'][0]-bb['test'][0]:+.3f}")

    print(f"\n  Random Forest:")
    print(f"    Train: R²={br['train'][0]:.3f}, MAE={br['train'][1]:.2f}")
    print(f"    Test:  R²={br['test'][0]:.3f}, MAE={br['test'][1]:.2f}")
    print(f"    Gap:   R²={br['train'][0]-br['test'][0]:+.3f}")

    elapsed = time.time() - t_start
    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
