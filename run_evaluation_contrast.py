"""
run_evaluation_contrast.py — Demonstrate how the SAME model produces different
accuracy metrics under different evaluation protocols.

Shows that the model CAN achieve high accuracy — the question is whether
the evaluation is honest.

Trains under 6 conditions:
  A. Leaky features + random split     (what many papers do)
  B. Leaky features + chrono split     (fixes split, not features)
  C. Clean features + random split     (fixes features, not split)
  D. Clean features + chrono split     (fully honest)
  E. GRACE features + chrono split     (GRACE honest)
  F. GRACE features + chrono + ES      (GRACE optimized)

For each: reports TRAIN R², TEST R², MAE, and gap.
"""

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# Rating hyperparameters
K0, V_GAIN, RD0, RD_MIN, C_DECAY, K_ITEM = 0.35, 0.04, 0.35, 0.06, 0.010, 0.02

PSEUDO = [
    'Номер теста', 'Длительность', 'Темы', 'Темы среза/дз (прошлый урок)',
    'Количественные характеристики', 'Ментор', 'Туториал', 'Время потока',
    'Оценка учителя', 'Школа', 'Отдел Менторов', 'Обратная связь получена',
    'КРА', 'МКЕ', 'Оценка среза', 'Баллы за тест', 'Попытки', 'Ранг',
    'Комментарий', 'Комментарий.', 'Срез Общее',
]

XGB = dict(
    n_estimators=1000, learning_rate=0.04, max_depth=10,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
    reg_lambda=1.5, reg_alpha=0.1, objective="reg:squarederror",
    tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0,
)


def load():
    df = pd.read_csv("report/student_performance_data.csv", low_memory=False)
    df = df[~df["skill_name"].isin(PSEUDO)].copy()
    df["test_datetime"] = pd.to_datetime(df["test_datetime"], errors="coerce")
    df = df.dropna(subset=["test_datetime"])
    df = df[df["test_datetime"] >= "2018-01-01"]
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["max_score"] = pd.to_numeric(df["max_score"], errors="coerce")
    df = df.dropna(subset=["score", "max_score"])
    df = df[(df["max_score"] > 0) & (df["score"] >= 0) & (df["score"] <= df["max_score"])]
    df = df.drop_duplicates()
    df = df[df["max_score"] > 2].copy()
    df["y"] = 100.0 * df["score"] / df["max_score"]
    return df


def build_leaky_features(df):
    """Features with leakage: rolling stats WITHOUT shift (include current score)."""
    df = df.sort_values(["student_id", "discipline", "test_datetime"]).copy()
    g = df.groupby(["student_id", "discipline"])["y"]
    # NO shift — current attempt's score leaks into its own features
    df["roll3"] = g.transform(lambda s: s.rolling(3, min_periods=1).mean().bfill().ffill())
    df["exp_avg"] = g.transform(lambda s: s.expanding().mean().bfill().ffill())
    df["lag1"] = g.transform(lambda s: s.shift(1).bfill().ffill())  # backfill = future leaks
    df["month"] = df["test_datetime"].dt.month
    df["dayofweek"] = df["test_datetime"].dt.dayofweek
    df["test_num"] = df.groupby(["student_id", "discipline"]).cumcount()
    for src, dst in [("discipline", "disc_code"), ("skill_name", "skill_code"),
                     ("test_type_category_id", "ttype_code")]:
        df[dst] = df[src].astype("category").cat.codes
    return df


def build_clean_features(df):
    """Leakage-free: shift(1) ensures no future information."""
    df = df.sort_values(["student_id", "discipline", "test_datetime"]).copy()
    g = df.groupby(["student_id", "discipline"])["y"]
    df["roll3"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["exp_avg"] = g.transform(lambda s: s.shift(1).expanding().mean())
    df["lag1"] = g.transform(lambda s: s.shift(1))
    df["month"] = df["test_datetime"].dt.month
    df["dayofweek"] = df["test_datetime"].dt.dayofweek
    df["test_num"] = df.groupby(["student_id", "discipline"]).cumcount()
    for src, dst in [("discipline", "disc_code"), ("skill_name", "skill_code"),
                     ("test_type_category_id", "ttype_code")]:
        df[dst] = df[src].astype("category").cat.codes
    return df


def build_grace_features(df):
    """Full GRACE rating features + aggregates."""
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
            elapsed = time.time() - t0
            print(f"      {t:,}/{n:,} ({100*t/n:.0f}%) [{elapsed:.0f}s]")

        i = students[t]
        s_key = (i, discs[t])
        sk = (i, skills[t])
        j, c = skills[t], ttypes[t]

        th_g = theta_g.get(i, 0.0)
        th_d = theta_d.get(s_key, th_g)
        th_s = theta_s.get(sk, th_d)
        dj = diff_skill.get(j, 0.0)
        dc = diff_ttype.get(c, 0.0)

        gap = ts[t] - last_seen.get(i, ts[t])
        v_g = min(rd_g_v.get(i, RD0**2) + C_DECAY * gap, RD0**2)
        v_d = min(rd_d_v.get(s_key, RD0**2) + C_DECAY * gap, RD0**2)

        e_hat = 1.0 / (1.0 + np.exp(-(0.5*th_d + 0.5*th_s - 0.7*dj - 0.3*dc)))
        sur = surprise.get(s_key, [])

        out[t] = [th_g, th_d, th_s, np.sqrt(v_g), np.sqrt(v_d), dj, dc, e_hat,
                  np.mean(sur[-5:]) if sur else 0.0, min(gap, 365.0),
                  n_g.get(i, 0), n_d.get(s_key, 0), n_s.get(sk, 0)]

        err = y[t] - e_hat
        k_g = K0 * v_g / (v_g + V_GAIN)
        k_d = K0 * v_d / (v_d + V_GAIN)
        theta_g[i] = th_g + 0.5 * k_g * err
        theta_d[s_key] = th_d + k_d * err
        theta_s[sk] = th_s + 1.5 * k_d * err
        diff_skill[j] = dj - K_ITEM * err
        diff_ttype[c] = dc - 0.5 * K_ITEM * err
        rd_g_v[i] = max(v_g * V_GAIN / (v_g + V_GAIN), RD_MIN**2)
        rd_d_v[s_key] = max(v_d * V_GAIN / (v_d + V_GAIN), RD_MIN**2)
        last_seen[i] = ts[t]
        n_g[i] = n_g.get(i, 0) + 1
        n_d[s_key] = n_d.get(s_key, 0) + 1
        n_s[sk] = n_s.get(sk, 0) + 1
        surprise[s_key] = (sur + [err])[-5:]

    print(f"      Rating loop done: {time.time()-t0:.1f}s")

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
    feats["max_score"] = df["max_score"].values
    feats["log_max_score"] = np.log1p(df["max_score"].values)
    feats["score_granularity"] = 100.0 / df["max_score"].values
    feats["ttype_id_code"] = df["test_type_id"].astype("category").cat.codes.values

    feats["y"] = y * 100.0
    feats["ts"] = dt.values
    feats["rid"] = rid
    feats["student_id"] = df["student_id"].values
    feats["discipline"] = df["discipline"].values

    # Add aggregate (clean)
    feats_s = feats.sort_values(["student_id", "discipline", "ts"]).copy()
    g = feats_s.groupby(["student_id", "discipline"])["y"]
    feats_s["lag1"] = g.transform(lambda s: s.shift(1))
    feats_s["roll3"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    feats_s["roll5"] = g.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    feats_s["exp_avg"] = g.transform(lambda s: s.shift(1).expanding().mean())
    gs = feats_s.groupby(["student_id", "skill_code"])["y"]
    feats_s["skill_lag1"] = gs.transform(lambda s: s.shift(1))
    feats_s["skill_roll3"] = gs.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    feats_s["test_num"] = feats_s.groupby(["student_id", "discipline"]).cumcount()
    feats_s["theta_range"] = feats_s["theta_g"] - feats_s["theta_disc"]
    feats_s["score_momentum"] = feats_s["surprise5"] * feats_s["n_disc"].clip(upper=20)
    feats_s["gap_bucket"] = pd.cut(feats_s["days_gap"], bins=[-1,0,1,3,7,14,30,365],
                                    labels=False).fillna(6).astype(int)
    feats_s["is_cold"] = (feats_s["n_disc"] < 3).astype(int)
    feats_s["ability_diff_gap"] = feats_s["theta_skill"] - feats_s["diff_skill"]

    return feats_s.sort_values("ts").reset_index(drop=True)


AGG_FEAT = ["roll3", "exp_avg", "lag1", "month", "dayofweek", "test_num",
            "disc_code", "skill_code", "ttype_code"]

GRACE_FEAT = [
    "theta_g", "theta_disc", "theta_skill", "rd_g", "rd_disc",
    "diff_skill", "diff_ttype", "e_hat", "surprise5", "days_gap",
    "n_global", "n_disc", "n_skill", "month", "dayofweek",
    "disc_code", "skill_code", "ttype_code",
    "max_score", "log_max_score", "score_granularity", "ttype_id_code",
    "roll3", "roll5", "exp_avg", "lag1", "skill_lag1", "skill_roll3",
    "test_num", "theta_range", "score_momentum", "gap_bucket",
    "is_cold", "ability_diff_gap",
]


def train_eval(X_tr, y_tr, X_te, y_te, early_stop=False, X_cal=None, y_cal=None):
    params = dict(XGB)
    if early_stop:
        params["n_estimators"] = 5000
        params["early_stopping_rounds"] = 50
    m = XGBRegressor(**params)
    if early_stop and X_cal is not None:
        m.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)], verbose=False)
        trees = m.best_iteration
    else:
        m.fit(X_tr, y_tr)
        trees = params["n_estimators"]

    p_tr = m.predict(X_tr)
    p_te = m.predict(X_te)
    r2_tr = r2_score(y_tr, p_tr)
    mae_tr = mean_absolute_error(y_tr, p_tr)
    r2_te = r2_score(y_te, p_te)
    mae_te = mean_absolute_error(y_te, p_te)
    return r2_tr, mae_tr, r2_te, mae_te, trees


def main():
    t0 = time.time()
    print("=" * 80)
    print("  EVALUATION PROTOCOL CONTRAST: Same Model, Different Metrics")
    print("=" * 80)

    print("\n  Loading data...")
    df = load()
    print(f"  Corpus: {len(df):,} rows")

    results = {}

    # ================================================================
    # A. LEAKY + RANDOM (what many papers do)
    # ================================================================
    print("\n  [A] Building LEAKY features + RANDOM split...")
    leaky = build_leaky_features(df)
    leaky_clean = leaky.dropna(subset=["exp_avg"]).copy()
    leaky_sorted = leaky_clean.sort_values("test_datetime").reset_index(drop=True)

    # Random split (ignoring time)
    tr_a, te_a = train_test_split(leaky_sorted, test_size=0.2, random_state=SEED)
    X_tr = tr_a[AGG_FEAT].fillna(-999)
    X_te = te_a[AGG_FEAT].fillna(-999)
    r2_tr, mae_tr, r2_te, mae_te, trees = train_eval(X_tr, tr_a["y"], X_te, te_a["y"])
    results["A. Leaky + Random"] = (r2_tr, mae_tr, r2_te, mae_te, len(tr_a), len(te_a), trees)
    print(f"      Train R2={r2_tr:.4f}  Test R2={r2_te:.4f}  MAE={mae_te:.2f}")

    # ================================================================
    # B. LEAKY + CHRONOLOGICAL
    # ================================================================
    print("\n  [B] LEAKY features + CHRONO split...")
    n = len(leaky_sorted)
    i80 = int(n * 0.8)
    tr_b, te_b = leaky_sorted.iloc[:i80], leaky_sorted.iloc[i80:]
    X_tr = tr_b[AGG_FEAT].fillna(-999)
    X_te = te_b[AGG_FEAT].fillna(-999)
    r2_tr, mae_tr, r2_te, mae_te, trees = train_eval(X_tr, tr_b["y"], X_te, te_b["y"])
    results["B. Leaky + Chrono"] = (r2_tr, mae_tr, r2_te, mae_te, len(tr_b), len(te_b), trees)
    print(f"      Train R2={r2_tr:.4f}  Test R2={r2_te:.4f}  MAE={mae_te:.2f}")

    # ================================================================
    # C. CLEAN + RANDOM
    # ================================================================
    print("\n  [C] CLEAN features + RANDOM split...")
    clean = build_clean_features(df)
    clean = clean.dropna(subset=["exp_avg"]).sort_values("test_datetime").reset_index(drop=True)

    tr_c, te_c = train_test_split(clean, test_size=0.2, random_state=SEED)
    X_tr = tr_c[AGG_FEAT].fillna(-999)
    X_te = te_c[AGG_FEAT].fillna(-999)
    r2_tr, mae_tr, r2_te, mae_te, trees = train_eval(X_tr, tr_c["y"], X_te, te_c["y"])
    results["C. Clean + Random"] = (r2_tr, mae_tr, r2_te, mae_te, len(tr_c), len(te_c), trees)
    print(f"      Train R2={r2_tr:.4f}  Test R2={r2_te:.4f}  MAE={mae_te:.2f}")

    # ================================================================
    # D. CLEAN + CHRONOLOGICAL (honest baseline)
    # ================================================================
    print("\n  [D] CLEAN features + CHRONO split (honest baseline)...")
    n = len(clean)
    i80 = int(n * 0.8)
    tr_d, te_d = clean.iloc[:i80], clean.iloc[i80:]
    X_tr = tr_d[AGG_FEAT].fillna(-999)
    X_te = te_d[AGG_FEAT].fillna(-999)
    r2_tr, mae_tr, r2_te, mae_te, trees = train_eval(X_tr, tr_d["y"], X_te, te_d["y"])
    results["D. Clean + Chrono"] = (r2_tr, mae_tr, r2_te, mae_te, len(tr_d), len(te_d), trees)
    print(f"      Train R2={r2_tr:.4f}  Test R2={r2_te:.4f}  MAE={mae_te:.2f}")

    # ================================================================
    # E. GRACE + CHRONOLOGICAL (no ES)
    # ================================================================
    print("\n  [E] GRACE features + CHRONO split...")
    print("    Building GRACE rating features...")
    grace = build_grace_features(df)
    gf = [f for f in GRACE_FEAT if f in grace.columns]
    n = len(grace)
    i70, i80 = int(n*0.7), int(n*0.8)
    tr_e, cal_e, te_e = grace.iloc[:i70], grace.iloc[i70:i80], grace.iloc[i80:]

    X_tr = tr_e[gf].fillna(-999)
    X_te = te_e[gf].fillna(-999)
    r2_tr, mae_tr, r2_te, mae_te, trees = train_eval(X_tr, tr_e["y"], X_te, te_e["y"])
    results["E. GRACE + Chrono"] = (r2_tr, mae_tr, r2_te, mae_te, len(tr_e), len(te_e), trees)
    print(f"      Train R2={r2_tr:.4f}  Test R2={r2_te:.4f}  MAE={mae_te:.2f}")

    # ================================================================
    # F. GRACE + CHRONO + EARLY STOPPING + TUNED (best honest result)
    # ================================================================
    print("\n  [F] GRACE + CHRONO + Early Stopping + Tuned...")
    X_cal = cal_e[gf].fillna(-999)
    r2_tr, mae_tr, r2_te, mae_te, trees = train_eval(
        X_tr, tr_e["y"], X_te, te_e["y"],
        early_stop=True, X_cal=X_cal, y_cal=cal_e["y"])
    results["F. GRACE + ES (best)"] = (r2_tr, mae_tr, r2_te, mae_te, len(tr_e), len(te_e), trees)
    print(f"      Train R2={r2_tr:.4f}  Test R2={r2_te:.4f}  MAE={mae_te:.2f}  trees={trees}")

    # ================================================================
    # FINAL TABLE
    # ================================================================
    print("\n" + "=" * 80)
    print("  COMPLETE RESULTS: HOW EVALUATION PROTOCOL CHANGES APPARENT ACCURACY")
    print("=" * 80)
    print(f"\n  {'Protocol':<28s} {'Train R2':>9s} {'Test R2':>9s} {'Test MAE':>9s} {'Gap':>7s} {'Trees':>6s} {'n_test':>10s}")
    print("  " + "-" * 82)

    for name, (r2_tr, mae_tr, r2_te, mae_te, ntr, nte, trees) in results.items():
        gap = r2_tr - r2_te
        print(f"  {name:<28s} {r2_tr:>9.4f} {r2_te:>9.4f} {mae_te:>9.2f} {gap:>+7.3f} {trees:>6d} {nte:>10,}")

    # Decompose the inflation
    a_te = results["A. Leaky + Random"][2]
    b_te = results["B. Leaky + Chrono"][2]
    c_te = results["C. Clean + Random"][2]
    d_te = results["D. Clean + Chrono"][2]
    e_te = results["E. GRACE + Chrono"][2]
    f_te = results["F. GRACE + ES (best)"][2]

    print(f"\n  {'INFLATION DECOMPOSITION':}")
    print(f"  " + "-" * 60)
    print(f"  Total inflation (A vs D):          {a_te - d_te:+.3f} R2")
    print(f"    Due to feature leakage:          {a_te - c_te:+.3f} R2  (leaky vs clean, same random split)")
    print(f"    Due to random split:             {c_te - d_te:+.3f} R2  (random vs chrono, same clean features)")
    print(f"  ")
    print(f"  GRACE improvement over baseline:   {f_te - d_te:+.3f} R2  (honest improvement)")
    print(f"  GRACE training accuracy:           {results['F. GRACE + ES (best)'][0]:.3f} R2")
    print(f"  GRACE test accuracy:               {f_te:.3f} R2")
    print(f"  Generalization gap:                {results['F. GRACE + ES (best)'][0] - f_te:+.3f} R2")

    print(f"\n  {'KEY INSIGHT':}")
    print(f"  " + "-" * 60)
    print(f"  The model LEARNS well (train R2 = {results['A. Leaky + Random'][0]:.3f} with leaky features)")
    print(f"  The model GENERALIZES honestly (test R2 = {f_te:.3f} under proper evaluation)")
    print(f"  The gap ({a_te:.3f} vs {f_te:.3f}) is NOT model weakness -")
    print(f"  it is the difference between honest and inflated evaluation.")
    print(f"  The {a_te - f_te:+.3f} R2 'accuracy' from leakage+random split")
    print(f"  exists only in evaluation, never in deployment.")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
