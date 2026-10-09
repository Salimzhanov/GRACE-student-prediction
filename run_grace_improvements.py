"""
run_grace_improvements.py — GRACE improvement experiments

I1. Optuna-tuned Glicko constants (11 params, configurable trials on 8k subsample)
I2. XGBoost early stopping + shallower trees (max_depth 8→6, ceiling 3000)
I3. Cross-discipline cold-start transfer (RD & theta_d initialization)

Cumulative evaluation:
  Baseline → +I2 → +I2+I3 → +I1+I2 → +I1+I2+I3

Usage:
    python run_grace_improvements.py           # full run (~60 min)
    python run_grace_improvements.py --fast    # 5k students, 15 Optuna trials (~8 min)
"""

import sys, time, json, warnings, argparse
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("WARNING: optuna not installed — I1 skipped. pip install optuna")

warnings.filterwarnings("ignore")
SEED  = 42
ALPHA = 0.10
np.random.seed(SEED)
OUT   = Path("results")
OUT.mkdir(exist_ok=True)

PSEUDO = [
    'Номер теста','Длительность','Темы','Темы среза/дз (прошлый урок)',
    'Количественные характеристики','Ментор','Туториал','Время потока',
    'Оценка учителя','Школа','Отдел Менторов','Обратная связь получена',
    'КРА','МКЕ','Оценка среза','Баллы за тест','Попытки','Ранг',
    'Комментарий','Комментарий.','Срез Общее',
]

# ── Published Glicko constants (baseline) ─────────────────────────────────────
BASELINE_GLICKO = dict(
    k0=0.35, v_gain=0.04, rd0=0.35, rd_min=0.06,
    c_decay=0.010, k_item=0.02, ema_halflife=90.0,
    w_theta_d=0.5, w_diff_skill=0.7,
    update_disc_mul=1.0, update_skill_mul=1.5,
)

# ── Feature groups (70 total, identical to run_evaluation_phases.py) ──────────
CORE = [
    "theta_g","theta_disc","theta_skill","rd_g","rd_disc","rd_skill",
    "diff_skill","diff_ttype","diff_skill_ema","e_hat","surprise5",
    "days_gap","days_gap_disc","days_gap_skill",
    "n_global","n_disc","n_skill",
    "month","dayofweek","disc_code","skill_code","ttype_code",
    "max_score","log_max_score","score_granularity","ttype_id_code",
]
GROUP_A = [
    "lag1","roll3","roll5","roll10","exp_avg","exp_std",
    "skill_lag1","skill_roll3","skill_exp_avg","skill_n",
    "test_num","trend_5v5","roll5_std",
    "theta_range","score_momentum","gap_bucket","gap_bucket_disc",
    "is_cold","is_cold_skill","e_hat_residual",
]
GROUP_B = [
    "ability_diff_gap",
    "student_tenure","disc_breadth","skill_breadth","intensity",
    "disc_mean_score","disc_std_score","skill_mean_score","skill_std_score",
    "theta_vs_disc",
    "month_sin","month_cos","dow_sin","dow_cos",
]
GROUP_C = [
    "global_lag1","global_roll3","global_roll5","global_roll10",
    "global_std5","ability_gap_global",
    "session_pos","session_score_ma",
    "days_to_june","in_exam_season",
]
ALL_FEATS = CORE + GROUP_A + GROUP_B + GROUP_C  # 70

# Features available without add_session_features / add_static_content_features
# Used in Optuna objective (speed-up: skip those two expensive steps)
_SKIP_OBJ = {"session_pos","session_score_ma",
             "disc_mean_score","disc_std_score",
             "skill_mean_score","skill_std_score","theta_vs_disc"}
OBJ_FEATS = [f for f in ALL_FEATS if f not in _SKIP_OBJ]  # 63 features


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_clean(fast_n=None):
    df = pd.read_csv("report/student_performance_data.csv", low_memory=False)
    df = df[~df["skill_name"].isin(PSEUDO)].copy()
    df["test_datetime"] = pd.to_datetime(df["test_datetime"], errors="coerce")
    df = df.dropna(subset=["test_datetime"])
    df = df[df["test_datetime"] >= "2018-01-01"]
    df["score"]     = pd.to_numeric(df["score"],     errors="coerce")
    df["max_score"] = pd.to_numeric(df["max_score"], errors="coerce")
    df = df.dropna(subset=["score","max_score"])
    df = df[(df["max_score"] > 2) & (df["score"] >= 0) & (df["score"] <= df["max_score"])]
    df = df.drop_duplicates()
    df["y"] = 100.0 * df["score"] / df["max_score"]
    if fast_n:
        rng  = np.random.default_rng(SEED)
        sids = df["student_id"].unique()
        keep = rng.choice(sids, min(fast_n, len(sids)), replace=False)
        df   = df[df["student_id"].isin(keep)].copy()
    return df.sort_values("test_datetime", kind="mergesort").reset_index(drop=True)


def compute_disc_offsets(df, n_train):
    """Per-discipline logit-scale difficulty offsets computed from training data only.
    offset[d] > 0 means discipline d is easier than average; < 0 means harder.
    Used to improve theta_d initialization for cold-start students.
    """
    df_tr  = df.iloc[:n_train]
    y_norm = (df_tr["score"] / df_tr["max_score"]).clip(0.01, 0.99)
    g_logit = float(np.log(y_norm.mean() / (1.0 - y_norm.mean())))
    disc_m  = df_tr.groupby("discipline").apply(
        lambda g: float(np.clip((g["score"] / g["max_score"]).mean(), 0.01, 0.99))
    )
    return {d: float(np.log(m / (1.0 - m))) - g_logit for d, m in disc_m.items()}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1: PARAMETRIC GLICKO RATING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def build_rating_features(df, gp: dict, disc_offsets: dict = None):
    """
    Streaming Glicko rating loop.

    gp: Glicko params dict (see BASELINE_GLICKO for keys).
    disc_offsets: dict[discipline -> logit offset] for I3 cold-start transfer.
                  Pass None to use baseline behaviour (theta_d init = theta_g).
    """
    k0        = gp["k0"]
    v_gain    = gp["v_gain"]
    rd0       = gp["rd0"]
    rd_min    = gp["rd_min"]
    c_decay   = gp["c_decay"]
    k_item    = gp["k_item"]
    ema_decay = np.log(2) / gp["ema_halflife"]
    w_td      = gp["w_theta_d"]
    w_ts      = 1.0 - w_td
    w_ds      = gp["w_diff_skill"]
    w_dt      = 1.0 - w_ds
    upd_disc  = gp.get("update_disc_mul",  1.0)
    upd_skill = gp.get("update_skill_mul", 1.5)
    use_xfer  = disc_offsets is not None

    n  = len(df)
    y  = (df["score"] / df["max_score"]).clip(0, 1).to_numpy()
    ts = pd.to_datetime(df["test_datetime"]).astype("int64").to_numpy() / 86.4e12

    students = df["student_id"].to_numpy()
    discs    = df["discipline"].astype(str).to_numpy()
    skills   = df["skill_name"].astype(str).to_numpy()
    ttypes   = df["test_type_category_id"].astype(str).to_numpy()

    theta_g, theta_d, theta_s                 = {}, {}, {}
    rd_g_v,  rd_d_v,  rd_s_v                 = {}, {}, {}
    diff_skill, diff_ttype, diff_skill_ema    = {}, {}, {}
    last_g,  last_d,  last_s                 = {}, {}, {}
    n_g,     n_d,     n_s                    = {}, {}, {}
    surprise, first_seen                     = {}, {}
    disc_sets, skill_sets, global_hist       = {}, {}, {}

    out = np.zeros((n, 27), dtype=np.float64)

    for t in range(n):
        i     = students[t]
        s_key = (i, discs[t])
        sk    = (i, skills[t])
        j     = skills[t]
        c     = ttypes[t]

        th_g = theta_g.get(i, 0.0)

        # I3: cold-start transfer — smarter theta_d & RD init for new disciplines
        if use_xfer and s_key not in theta_d:
            n_other  = len(disc_sets.get(i, set()))
            # Transfer: reduce initial RD based on cross-discipline experience
            # xfer_f ∈ [0.30, 1.0]; experienced students get tighter prior
            xfer_f   = max(0.30, np.exp(-0.10 * n_other))
            v_d_init = rd0 ** 2 * xfer_f
            # Disc-specific ability prior: adjust theta_d for known discipline difficulty
            th_d_init = th_g + disc_offsets.get(discs[t], 0.0)
        else:
            v_d_init  = rd0 ** 2
            th_d_init = theta_d.get(s_key, th_g)

        th_d = theta_d.get(s_key, th_d_init)
        th_s = theta_s.get(sk, th_d)
        dj   = diff_skill.get(j, 0.0)
        dc   = diff_ttype.get(c, 0.0)

        gap_g = ts[t] - last_g.get(i,     ts[t])
        gap_d = ts[t] - last_d.get(s_key, ts[t])
        gap_s = ts[t] - last_s.get(sk,    ts[t])

        v_g = min(rd_g_v.get(i,     rd0**2)   + c_decay * gap_g, rd0**2)
        v_d = min(rd_d_v.get(s_key, v_d_init) + c_decay * gap_d, rd0**2)
        v_s = min(rd_s_v.get(sk,    rd0**2)   + c_decay * gap_s, rd0**2)

        d_ema = diff_skill_ema.get(j, 0.0) * np.exp(-gap_s * ema_decay)
        e_hat = 1.0 / (1.0 + np.exp(
            -(w_td*th_d + w_ts*th_s - w_ds*(dj + d_ema) - w_dt*dc)
        ))
        sur = surprise.get(s_key, [])

        first_t   = first_seen.get(i, ts[t])
        tenure    = ts[t] - first_t
        n_g_val   = n_g.get(i, 0)
        intensity = n_g_val / max(tenure, 1.0)

        hist   = global_hist.get(i, deque(maxlen=10))
        h_list = list(hist)
        if h_list:
            g_lag1   = h_list[-1]           * 100.0
            g_roll3  = np.mean(h_list[-3:]) * 100.0
            g_roll5  = np.mean(h_list[-5:]) * 100.0
            g_roll10 = np.mean(h_list)      * 100.0
            g_std5   = (np.std(h_list[-5:]) * 100.0) if len(h_list) >= 2 else 0.0
            g_gap    = e_hat * 100.0 - g_roll5
        else:
            g_lag1 = g_roll3 = g_roll5 = g_roll10 = e_hat * 100.0
            g_std5 = g_gap = 0.0

        out[t] = [
            th_g, th_d, th_s,
            np.sqrt(v_g), np.sqrt(v_d), np.sqrt(v_s),
            dj, dc, d_ema, e_hat,
            np.mean(sur[-5:]) if sur else 0.0,
            min(gap_g, 365.0), min(gap_d, 365.0), min(gap_s, 365.0),
            float(n_g_val), float(n_d.get(s_key, 0)), float(n_s.get(sk, 0)),
            min(tenure, 3650.0),
            float(len(disc_sets.get(i, set()))),
            float(len(skill_sets.get(i, set()))),
            min(intensity, 20.0),
            g_lag1, g_roll3, g_roll5, g_roll10, g_std5, g_gap,
        ]

        # ── state updates ────────────────────────────────────────────────────
        err = y[t] - e_hat
        k_g = k0 * v_g / (v_g + v_gain)
        k_d = k0 * v_d / (v_d + v_gain)
        k_s = k0 * v_s / (v_s + v_gain)
        theta_g[i]     = th_g + 0.5       * k_g * err
        theta_d[s_key] = th_d + upd_disc  * k_d * err
        theta_s[sk]    = th_s + upd_skill * k_s * err
        diff_skill[j]  = dj   - k_item         * err
        diff_ttype[c]  = dc   - 0.5 * k_item   * err
        diff_skill_ema[j] = d_ema - k_item      * err
        rd_g_v[i]      = max(v_g * v_gain / (v_g + v_gain), rd_min**2)
        rd_d_v[s_key]  = max(v_d * v_gain / (v_d + v_gain), rd_min**2)
        rd_s_v[sk]     = max(v_s * v_gain / (v_s + v_gain), rd_min**2)
        last_g[i] = last_d[s_key] = last_s[sk] = ts[t]
        n_g[i]     = n_g_val + 1
        n_d[s_key] = n_d.get(s_key, 0) + 1
        n_s[sk]    = n_s.get(sk,    0) + 1
        surprise[s_key] = (sur + [err])[-5:]
        if i not in first_seen:
            first_seen[i] = ts[t]
        disc_sets.setdefault(i, set()).add(discs[t])
        skill_sets.setdefault(i, set()).add(skills[t])
        hist.append(y[t])
        global_hist[i] = hist

    rcols = [
        "theta_g","theta_disc","theta_skill",
        "rd_g","rd_disc","rd_skill",
        "diff_skill","diff_ttype","diff_skill_ema","e_hat","surprise5",
        "days_gap","days_gap_disc","days_gap_skill",
        "n_global","n_disc","n_skill",
        "student_tenure","disc_breadth","skill_breadth","intensity",
        "global_lag1","global_roll3","global_roll5","global_roll10",
        "global_std5","ability_gap_global",
    ]
    feats = pd.DataFrame(out, columns=rcols)
    dt = pd.to_datetime(df["test_datetime"])
    feats["month"]     = dt.dt.month.values
    feats["dayofweek"] = dt.dt.dayofweek.values
    for src, col in [("discipline","disc_code"), ("skill_name","skill_code"),
                     ("test_type_category_id","ttype_code")]:
        feats[col] = df[src].astype("category").cat.codes.values
    feats["max_score"]         = df["max_score"].values
    feats["log_max_score"]     = np.log1p(df["max_score"].values)
    feats["score_granularity"] = 100.0 / df["max_score"].values
    feats["ttype_id_code"]     = (df["test_type_id"].astype("category").cat.codes.values
                                  if "test_type_id" in df.columns else 0)
    feats["y"]          = df["y"].values
    feats["ts"]         = dt.values
    feats["student_id"] = df["student_id"].values
    feats["discipline"] = df["discipline"].values
    feats["skill_name"] = df["skill_name"].values
    return feats


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (non-rating layers; unchanged from baseline)
# ══════════════════════════════════════════════════════════════════════════════

def build_aggregate_features(feats):
    df = feats.sort_values(["student_id","discipline","ts"]).copy()
    g  = df.groupby(["student_id","discipline"])["y"]
    df["lag1"]      = g.transform(lambda s: s.shift(1))
    df["roll3"]     = g.transform(lambda s: s.shift(1).rolling(3,  min_periods=1).mean())
    df["roll5"]     = g.transform(lambda s: s.shift(1).rolling(5,  min_periods=1).mean())
    df["roll10"]    = g.transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    df["exp_avg"]   = g.transform(lambda s: s.shift(1).expanding().mean())
    df["exp_std"]   = g.transform(lambda s: s.shift(1).expanding().std())
    df["roll5_std"] = g.transform(lambda s: s.shift(1).rolling(5, min_periods=2).std())
    roll5_first     = g.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean().shift(4))
    df["trend_5v5"] = df["roll5"] - roll5_first
    df["test_num"]  = df.groupby(["student_id","discipline"]).cumcount()
    gs = df.groupby(["student_id","skill_name"])["y"]
    df["skill_lag1"]    = gs.transform(lambda s: s.shift(1))
    df["skill_roll3"]   = gs.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["skill_exp_avg"] = gs.transform(lambda s: s.shift(1).expanding().mean())
    df["skill_n"]       = gs.transform("cumcount")
    return df.sort_values("ts").reset_index(drop=True)


def add_derived_features(df):
    df = df.copy()
    df["theta_range"]      = df["theta_g"] - df["theta_disc"]
    df["score_momentum"]   = df["surprise5"] * df["n_disc"].clip(upper=20)
    df["gap_bucket"]       = pd.cut(df["days_gap"],
                                    bins=[-1,0,1,3,7,14,30,365], labels=False
                                    ).fillna(6).astype(int)
    df["gap_bucket_disc"]  = pd.cut(df["days_gap_disc"],
                                    bins=[-1,0,1,3,7,14,30,365], labels=False
                                    ).fillna(6).astype(int)
    df["is_cold"]          = (df["n_disc"]  < 3).astype(int)
    df["is_cold_skill"]    = (df["n_skill"] < 2).astype(int)
    df["e_hat_residual"]   = df["e_hat"]*100 - df["exp_avg"].fillna(df["e_hat"]*100)
    df["ability_diff_gap"] = df["theta_skill"] - df["diff_skill"]
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    df["dow_sin"]   = np.sin(2*np.pi*df["dayofweek"]/7)
    df["dow_cos"]   = np.cos(2*np.pi*df["dayofweek"]/7)
    return df


def add_session_features(feats):
    SESSION_GAP_S = 3600.0
    df = feats.sort_values(["student_id","ts"]).copy()
    ts_s    = pd.to_datetime(df["ts"]).astype("int64").values / 1e9
    sid_arr = df["student_id"].values
    new_stu = np.concatenate([[True], sid_arr[1:] != sid_arr[:-1]])
    gap_arr = np.concatenate([[0.0], ts_s[1:] - ts_s[:-1]])
    df["_new_sess"] = (new_stu | (gap_arr > SESSION_GAP_S)).astype(int)
    df["_sess_id"]  = df.groupby("student_id")["_new_sess"].cumsum()
    df["session_pos"]      = df.groupby(["student_id","_sess_id"]).cumcount()
    df["session_score_ma"] = df.groupby(["student_id","_sess_id"])["y"].transform(
        lambda s: s.shift(1).expanding().mean())
    df = df.sort_index()
    feats = feats.copy()
    feats["session_pos"]      = df["session_pos"].values
    feats["session_score_ma"] = df["session_score_ma"].values
    return feats


def add_exam_features(feats):
    dt    = pd.to_datetime(feats["ts"])
    june1 = pd.to_datetime(dt.dt.year.astype(str) + "-06-01", format="%Y-%m-%d")
    feats = feats.copy()
    feats["days_to_june"]   = (june1 - dt).dt.days.clip(0, 365).astype(float).values
    feats["in_exam_season"] = ((dt.dt.month >= 2) & (dt.dt.month <= 5)).astype(float).values
    return feats


def add_static_content_features(feats_all, train_mask):
    drop = ["disc_mean_score","disc_std_score","skill_mean_score","skill_std_score","theta_vs_disc"]
    feats_all = feats_all.drop(columns=[c for c in drop if c in feats_all.columns])
    tr = feats_all[train_mask]
    g_mean, g_std = tr["y"].mean(), tr["y"].std()
    disc_agg  = tr.groupby("discipline")["y"].agg(
        disc_mean_score="mean", disc_std_score="std").reset_index()
    skill_agg = tr.groupby("skill_name")["y"].agg(
        skill_mean_score="mean", skill_std_score="std").reset_index()
    feats_all = feats_all.merge(disc_agg,  on="discipline",  how="left")
    feats_all = feats_all.merge(skill_agg, on="skill_name",  how="left")
    for col, fill in [("disc_mean_score",g_mean), ("disc_std_score",g_std),
                      ("skill_mean_score",g_mean), ("skill_std_score",g_std)]:
        feats_all[col] = feats_all[col].fillna(fill)
    feats_all["theta_vs_disc"] = feats_all["theta_skill"] - feats_all["disc_mean_score"] / 100.0
    return feats_all


def chrono_split(feats, train_f=0.70, cal_f=0.10):
    n  = len(feats)
    i1 = int(n * train_f)
    i2 = int(n * (train_f + cal_f))
    return feats.iloc[:i1].copy(), feats.iloc[i1:i2].copy(), feats.iloc[i2:].copy()


def temporal_weights(ts_series, halflife=730.0):
    dt  = pd.to_datetime(ts_series)
    age = (dt.max() - dt).dt.days.values.astype(float)
    w   = np.exp(-np.log(2.0) * age / halflife)
    return w / w.mean()


def full_pipeline(df, gp, disc_offsets=None):
    """Rating loop + all feature layers."""
    feats = build_rating_features(df, gp, disc_offsets=disc_offsets)
    feats = build_aggregate_features(feats)
    feats = add_derived_features(feats)
    feats = add_session_features(feats)
    feats = add_exam_features(feats)
    return feats


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2: XGBOOST (I2 — early stopping + max_depth=6)
# ══════════════════════════════════════════════════════════════════════════════

_XGB_BASE = dict(
    learning_rate=0.04, subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
    reg_lambda=2.0, reg_alpha=0.2, objective="reg:squarederror",
    tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0,
)

def xgb_fit(X_tr, y_tr, X_val=None, y_val=None, sw=None, early_stop=False, depth=8):
    """Train XGBoost with optional early stopping (I2)."""
    if early_stop and X_val is not None:
        params = {**_XGB_BASE, "n_estimators": 3000, "max_depth": depth,
                  "early_stopping_rounds": 50}
        m = XGBRegressor(**params)
        m.fit(X_tr, np.asarray(y_tr, dtype=np.float64),
              eval_set=[(X_val, np.asarray(y_val, dtype=np.float64))],
              sample_weight=sw, verbose=False)
    else:
        params = {**_XGB_BASE, "n_estimators": 1500, "max_depth": depth}
        m = XGBRegressor(**params)
        m.fit(X_tr, np.asarray(y_tr, dtype=np.float64), sample_weight=sw, verbose=False)
    return m


def best_n_trees(model, early_stop):
    if early_stop and hasattr(model, "best_iteration") and model.best_iteration is not None:
        return model.best_iteration + 1
    return _XGB_BASE.get("n_estimators", 1500)


# ══════════════════════════════════════════════════════════════════════════════
# I1: OPTUNA GLICKO TUNING
# ══════════════════════════════════════════════════════════════════════════════

def _optuna_objective(trial, df_sub):
    gp = dict(
        k0           = trial.suggest_float("k0",           0.10, 0.70),
        v_gain       = trial.suggest_float("v_gain",       0.01, 0.12),
        rd0          = trial.suggest_float("rd0",          0.15, 0.55),
        rd_min       = trial.suggest_float("rd_min",       0.02, 0.12),
        c_decay      = trial.suggest_float("c_decay",      0.002, 0.04, log=True),
        k_item       = trial.suggest_float("k_item",       0.005, 0.06, log=True),
        ema_halflife = trial.suggest_float("ema_halflife", 20.0, 180.0),
        w_theta_d    = trial.suggest_float("w_theta_d",    0.25, 0.75),
        w_diff_skill = trial.suggest_float("w_diff_skill", 0.40, 0.90),
        update_disc_mul  = trial.suggest_float("update_disc_mul",  0.5,  2.5),
        update_skill_mul = trial.suggest_float("update_skill_mul", 0.8,  3.5),
    )
    try:
        feats = build_rating_features(df_sub, gp)          # no session/static for speed
        feats = build_aggregate_features(feats)
        feats = add_derived_features(feats)
        feats = add_exam_features(feats)
        n  = len(feats)
        tr = feats.iloc[:int(0.75 * n)]
        te = feats.iloc[int(0.75 * n):]
        avail = [f for f in OBJ_FEATS if f in feats.columns]
        X_tr  = tr[avail].fillna(0).values
        X_te  = te[avail].fillna(0).values
        FAST  = dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                     subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                     tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0)
        m = XGBRegressor(**FAST)
        m.fit(X_tr, tr["y"].values, verbose=False)
        return float(r2_score(te["y"].values, m.predict(X_te)))
    except Exception:
        return -1.0


def run_optuna_tuning(df_sub, n_trials=50):
    print(f"\n  [I1] Optuna: {n_trials} trials on {df_sub['student_id'].nunique():,} students ...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    t0 = time.time()
    study.optimize(lambda trial: _optuna_objective(trial, df_sub),
                   n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - t0
    best = study.best_params
    print(f"  [I1] Best R² (subsample, 63-feat): {study.best_value:.4f}  [{elapsed:.0f}s]")
    print(f"  [I1] Best params:")
    for k, v in best.items():
        baseline_v = BASELINE_GLICKO.get(k, "—")
        print(f"         {k:<22} {v:.5g}   (was {baseline_v})")
    return best


# ══════════════════════════════════════════════════════════════════════════════
# FULL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(df, gp, label, early_stop=False, disc_offsets=None):
    """70/10/20 chrono evaluation with the given configuration."""
    t0 = time.time()
    print(f"\n  [{label}]")
    feats = full_pipeline(df, gp, disc_offsets=disc_offsets)

    n1 = int(len(feats) * 0.70)
    n2 = int(len(feats) * 0.80)

    train_mask = np.zeros(len(feats), dtype=bool)
    train_mask[:n1] = True
    feats = add_static_content_features(feats, train_mask)

    train = feats.iloc[:n1].copy()
    cal   = feats.iloc[n1:n2].copy()
    test  = feats.iloc[n2:].copy()

    avail  = [f for f in ALL_FEATS if f in feats.columns]
    X_tr   = train[avail].fillna(0).values
    X_ca   = cal[avail].fillna(0).values
    X_te   = test[avail].fillna(0).values
    y_tr, y_ca, y_te = train["y"].values, cal["y"].values, test["y"].values

    sw    = temporal_weights(train["ts"])
    depth = 6 if early_stop else 8
    model = xgb_fit(X_tr, y_tr, X_ca, y_ca, sw=sw, early_stop=early_stop, depth=depth)
    n_trees = best_n_trees(model, early_stop)

    p_tr = model.predict(X_tr)
    p_te = model.predict(X_te)
    r2_tr  = float(r2_score(y_tr, p_tr))
    r2_te  = float(r2_score(y_te, p_te))
    mae_te = float(mean_absolute_error(y_te, p_te))
    gap    = r2_tr - r2_te

    # Cold-start stratum (1–4 prior disc attempts)
    cold_m = test["n_disc"] < 5
    r2_cold = (float(r2_score(test.loc[cold_m,"y"], p_te[cold_m.values]))
               if cold_m.sum() > 100 else float("nan"))

    # History depth breakdown
    depth_r2 = {}
    for lbl, lo, hi in [("1-4",0,4),("5-10",5,10),("11-20",11,20),(">20",21,9999)]:
        m = (test["n_disc"] > lo) & (test["n_disc"] <= hi)
        if m.sum() > 100:
            depth_r2[lbl] = round(float(r2_score(test.loc[m,"y"], p_te[m.values])), 4)

    elapsed = time.time() - t0
    print(f"    R²={r2_te:.4f}  MAE={mae_te:.3f}  gen_gap={gap:.4f}  "
          f"R²_cold={r2_cold:.4f}  n_trees={n_trees}  [{elapsed:.0f}s]")
    print(f"    depth-R²: " +
          "  ".join(f"{k}:{v:.4f}" for k, v in depth_r2.items()))

    return dict(
        label=label,
        r2_test=round(r2_te, 4), r2_train=round(r2_tr, 4),
        mae_test=round(mae_te, 3), gen_gap=round(gap, 4),
        r2_cold=round(r2_cold, 4) if not np.isnan(r2_cold) else None,
        depth_r2=depth_r2, n_trees=n_trees,
        elapsed_s=round(elapsed, 1),
        glicko_params=gp, early_stop=early_stop,
        disc_transfer=(disc_offsets is not None),
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="5k students, 15 Optuna trials (~8 min)")
    args = parser.parse_args()

    fast_n     = 5_000 if args.fast else None
    opt_n_stu  = 5_000 if args.fast else 8_000
    opt_trials = 15    if args.fast else 50

    print("=" * 70)
    print("GRACE Improvement Experiments  (I1=Optuna · I2=EarlyStop · I3=ColdXfer)")
    print(f"Mode: {'FAST (5k students)' if args.fast else 'FULL (55k students)'}")
    print("=" * 70)

    t_total = time.time()

    print("\nLoading and cleaning data ...")
    df = load_and_clean(fast_n=fast_n)
    print(f"  {len(df):,} rows, {df['student_id'].nunique():,} students")

    # Disc offsets from training portion (for I3, computed once)
    n_train   = int(len(df) * 0.70)
    disc_off  = compute_disc_offsets(df, n_train)
    print(f"  Disc offsets: {len(disc_off)} disciplines "
          f"(range [{min(disc_off.values()):.2f}, {max(disc_off.values()):.2f}])")

    results = []

    # ── Baseline ──────────────────────────────────────────────────────────────
    results.append(evaluate(df, BASELINE_GLICKO, "Baseline", early_stop=False))

    # ── I2: Early stopping + max_depth=6 ──────────────────────────────────────
    results.append(evaluate(df, BASELINE_GLICKO, "I2: EarlyStop depth=6", early_stop=True))

    # ── I2 + I3: Add cold-start transfer ──────────────────────────────────────
    results.append(evaluate(df, BASELINE_GLICKO, "I2+I3: EarlyStop+ColdXfer",
                            early_stop=True, disc_offsets=disc_off))

    # ── I1: Optuna Glicko tuning (on subsample) ────────────────────────────────
    best_gp = None
    if HAS_OPTUNA:
        rng    = np.random.default_rng(SEED)
        sids   = df["student_id"].unique()
        keep   = rng.choice(sids, min(opt_n_stu, len(sids)), replace=False)
        df_sub = df[df["student_id"].isin(keep)].copy()
        tuned  = run_optuna_tuning(df_sub, n_trials=opt_trials)
        best_gp = {**BASELINE_GLICKO, **tuned}

        # I1 + I2
        results.append(evaluate(df, best_gp, "I1+I2: OptGlicko+EarlyStop",
                                early_stop=True))

        # I1 + I2 + I3 (all improvements)
        results.append(evaluate(df, best_gp, "I1+I2+I3: All improvements",
                                early_stop=True, disc_offsets=disc_off))
    else:
        print("\n  [I1] Skipped — install optuna: pip install optuna")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 82)
    print(f"  {'Config':<32}  {'R2':>6}  {'dR2':>6}  {'MAE':>6}  {'gap':>6}  "
          f"{'R2_cold':>8}  {'trees':>6}")
    print("  " + "-" * 78)
    base_r2 = results[0]["r2_test"]
    for r in results:
        delta = r["r2_test"] - base_r2
        cold  = f"{r['r2_cold']:.4f}" if r["r2_cold"] is not None else "   n/a"
        print(f"  {r['label']:<32}  {r['r2_test']:>6.4f}  {delta:>+6.4f}  "
              f"{r['mae_test']:>6.3f}  {r['gen_gap']:>6.4f}  {cold:>8}  {r['n_trees']:>6}")
    print("=" * 82)

    # Save results
    out_path = OUT / "improvements.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results -> {out_path}")
    print(f"  Total runtime: {(time.time() - t_total) / 60:.1f} min")


if __name__ == "__main__":
    main()
