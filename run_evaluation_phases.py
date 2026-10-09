"""
run_evaluation_phases.py — 10-phase comprehensive GRACE evaluation.

Phases
------
P1   Data audit            — corpus statistics, year/discipline breakdown
P2   Leakage audit         — 4 evaluation protocols showing inflation
P3   Baseline models       — Ridge, RF, XGBoost (same features/split)
P4   GRACE accuracy        — R², MAE, train-test gap, per-student R²
P5   Rolling-origin        — 5-fold expanding-window temporal validation
P6   History depth         — performance by prior-test-count stratum
P7   Feature ablation      — progressive group removal (C → B+C → A+B+C)
P8   Conformal intervals   — Mondrian CQR + Asymmetric CQR coverage/width
P9   Per-discipline        — R² and delta for each subject
P10  Bootstrap significance — paired residual bootstrap (B=2000)

Usage
-----
    python run_evaluation_phases.py           # full run (~90 min)
    python run_evaluation_phases.py --fast    # 5k-student subsample (~8 min)
"""

import sys, io, time, json, warnings, argparse
from pathlib import Path
from collections import deque

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
SEED  = 42
ALPHA = 0.10
np.random.seed(SEED)

# ── Glicko constants ──────────────────────────────────────────────────────────
K0, V_GAIN, RD0, RD_MIN, C_DECAY, K_ITEM = 0.35, 0.04, 0.35, 0.06, 0.010, 0.02
EMA_HALFLIFE    = 90.0
SESSION_GAP_S   = 3600.0
WEIGHT_HALFLIFE = 730.0

PSEUDO = [
    'Номер теста','Длительность','Темы','Темы среза/дз (прошлый урок)',
    'Количественные характеристики','Ментор','Туториал','Время потока',
    'Оценка учителя','Школа','Отдел Менторов','Обратная связь получена',
    'КРА','МКЕ','Оценка среза','Баллы за тест','Попытки','Ранг',
    'Комментарий','Комментарий.','Срез Общее',
]

OUT = Path("results")
OUT.mkdir(exist_ok=True)
LOG_PATH = OUT / "eval_phases_log.txt"
RES_PATH = OUT / "evaluation_phases.json"

_log_lines: list[str] = []
_results:   dict      = {}

def log(msg=""):
    print(msg, flush=True)
    _log_lines.append(str(msg))

def hdr(title):
    sep = "=" * 70
    log(f"\n{sep}")
    log(title)
    log(sep)

def rec(phase, key, val):
    _results.setdefault(phase, {})[key] = val
    log(f"  {key} = {val}")

def flush_outputs():
    LOG_PATH.write_text("\n".join(_log_lines), encoding="utf-8")
    RES_PATH.write_text(json.dumps(_results, indent=2, default=str), encoding="utf-8")


# ── Feature group definitions (70 total) ─────────────────────────────────────
CORE = [
    "theta_g","theta_disc","theta_skill","rd_g","rd_disc","rd_skill",
    "diff_skill","diff_ttype","diff_skill_ema","e_hat","surprise5",
    "days_gap","days_gap_disc","days_gap_skill",
    "n_global","n_disc","n_skill",
    "month","dayofweek","disc_code","skill_code","ttype_code",
    "max_score","log_max_score","score_granularity","ttype_id_code",
]  # 26

GROUP_A = [
    "lag1","roll3","roll5","roll10","exp_avg","exp_std",
    "skill_lag1","skill_roll3","skill_exp_avg","skill_n",
    "test_num","trend_5v5","roll5_std",
    "theta_range","score_momentum","gap_bucket","gap_bucket_disc",
    "is_cold","is_cold_skill","e_hat_residual",
]  # 20

GROUP_B = [
    "ability_diff_gap",
    "student_tenure","disc_breadth","skill_breadth","intensity",
    "disc_mean_score","disc_std_score","skill_mean_score","skill_std_score",
    "theta_vs_disc",
    "month_sin","month_cos","dow_sin","dow_cos",
]  # 14

GROUP_C = [
    "global_lag1","global_roll3","global_roll5","global_roll10",
    "global_std5","ability_gap_global",
    "session_pos","session_score_ma",
    "days_to_june","in_exam_season",
]  # 10

ALL_FEATS = CORE + GROUP_A + GROUP_B + GROUP_C  # 70

XGB_PARAMS = dict(
    n_estimators=1500, learning_rate=0.04, max_depth=8,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
    reg_lambda=2.0, reg_alpha=0.2, objective="reg:squarederror",
    tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0,
)
XGB_FAST = dict(
    n_estimators=400, learning_rate=0.05, max_depth=8,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
    reg_lambda=2.0, reg_alpha=0.2, objective="reg:squarederror",
    tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0,
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_clean(fast_n=None):
    hdr("DATA LOADING & CLEANING")
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
        log(f"  [FAST] subsampled to {len(keep):,} students")
    log(f"  Corpus: {len(df):,} rows, {df['student_id'].nunique():,} students")
    return df.sort_values("test_datetime", kind="mergesort").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def build_rating_features(df):
    """Glicko streaming rating loop — outputs 27 columns."""
    t0 = time.time()
    n  = len(df)
    y       = (df["score"] / df["max_score"]).clip(0, 1).to_numpy()
    ts      = pd.to_datetime(df["test_datetime"]).astype("int64").to_numpy() / 86.4e12
    students= df["student_id"].to_numpy()
    discs   = df["discipline"].astype(str).to_numpy()
    skills  = df["skill_name"].astype(str).to_numpy()
    ttypes  = df["test_type_category_id"].astype(str).to_numpy()

    theta_g, theta_d, theta_s = {}, {}, {}
    rd_g_v,  rd_d_v,  rd_s_v  = {}, {}, {}
    diff_skill, diff_ttype, diff_skill_ema = {}, {}, {}
    last_g,  last_d,  last_s   = {}, {}, {}
    n_g,     n_d,     n_s      = {}, {}, {}
    surprise     = {}
    first_seen   = {}
    disc_sets    = {}
    skill_sets   = {}
    global_hist  = {}
    EMA_DECAY = np.log(2) / EMA_HALFLIFE

    out = np.zeros((n, 27), dtype=np.float64)

    for t in range(n):
        if t % 500_000 == 0 and t > 0:
            log(f"    {t:,}/{n:,} ({100*t/n:.0f}%) [{time.time()-t0:.0f}s]")
        i     = students[t]
        s_key = (i, discs[t])
        sk    = (i, skills[t])
        j     = skills[t]
        c     = ttypes[t]

        th_g = theta_g.get(i,     0.0)
        th_d = theta_d.get(s_key, th_g)
        th_s = theta_s.get(sk,    th_d)
        dj   = diff_skill.get(j,  0.0)
        dc   = diff_ttype.get(c,  0.0)

        gap_g = ts[t] - last_g.get(i,     ts[t])
        gap_d = ts[t] - last_d.get(s_key, ts[t])
        gap_s = ts[t] - last_s.get(sk,    ts[t])

        v_g = min(rd_g_v.get(i,     RD0**2) + C_DECAY * gap_g, RD0**2)
        v_d = min(rd_d_v.get(s_key, RD0**2) + C_DECAY * gap_d, RD0**2)
        v_s = min(rd_s_v.get(sk,    RD0**2) + C_DECAY * gap_s, RD0**2)

        d_ema = diff_skill_ema.get(j, 0.0) * np.exp(-gap_s * EMA_DECAY)
        e_hat = 1.0 / (1.0 + np.exp(-(0.5*th_d + 0.5*th_s - 0.7*(dj + d_ema) - 0.3*dc)))
        sur   = surprise.get(s_key, [])

        first_t   = first_seen.get(i, ts[t])
        tenure    = ts[t] - first_t
        n_g_val   = n_g.get(i, 0)
        intensity = n_g_val / max(tenure, 1.0)

        hist   = global_hist.get(i, deque(maxlen=10))
        h_list = list(hist)
        if h_list:
            g_lag1   = h_list[-1]  * 100.0
            g_roll3  = np.mean(h_list[-3:]) * 100.0
            g_roll5  = np.mean(h_list[-5:]) * 100.0
            g_roll10 = np.mean(h_list)      * 100.0
            g_std5   = np.std(h_list[-5:])  * 100.0 if len(h_list) >= 2 else 0.0
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

        # State updates
        err = y[t] - e_hat
        k_g = K0 * v_g / (v_g + V_GAIN)
        k_d = K0 * v_d / (v_d + V_GAIN)
        k_s = K0 * v_s / (v_s + V_GAIN)
        theta_g[i]     = th_g + 0.5 * k_g * err
        theta_d[s_key] = th_d + 1.0 * k_d * err
        theta_s[sk]    = th_s + 1.5 * k_s * err
        diff_skill[j]  = dj   - K_ITEM * err
        diff_ttype[c]  = dc   - 0.5 * K_ITEM * err
        diff_skill_ema[j] = d_ema - K_ITEM * err
        rd_g_v[i]      = max(v_g * V_GAIN / (v_g + V_GAIN), RD_MIN**2)
        rd_d_v[s_key]  = max(v_d * V_GAIN / (v_d + V_GAIN), RD_MIN**2)
        rd_s_v[sk]     = max(v_s * V_GAIN / (v_s + V_GAIN), RD_MIN**2)
        last_g[i] = last_d[s_key] = last_s[sk] = ts[t]
        n_g[i]    = n_g_val + 1
        n_d[s_key] = n_d.get(s_key, 0) + 1
        n_s[sk]    = n_s.get(sk,    0) + 1
        surprise[s_key] = (sur + [err])[-5:]
        if i not in first_seen:
            first_seen[i] = ts[t]
        disc_sets.setdefault(i, set()).add(discs[t])
        skill_sets.setdefault(i, set()).add(skills[t])
        hist.append(y[t])
        global_hist[i] = hist

    log(f"  Rating loop: {time.time()-t0:.1f}s")
    cols = [
        "theta_g","theta_disc","theta_skill",
        "rd_g","rd_disc","rd_skill",
        "diff_skill","diff_ttype","diff_skill_ema","e_hat","surprise5",
        "days_gap","days_gap_disc","days_gap_skill",
        "n_global","n_disc","n_skill",
        "student_tenure","disc_breadth","skill_breadth","intensity",
        "global_lag1","global_roll3","global_roll5","global_roll10",
        "global_std5","ability_gap_global",
    ]
    feats = pd.DataFrame(out, columns=cols)
    dt = pd.to_datetime(df["test_datetime"])
    feats["month"]     = dt.dt.month.values
    feats["dayofweek"] = dt.dt.dayofweek.values
    for src, col in [("discipline","disc_code"),("skill_name","skill_code"),
                     ("test_type_category_id","ttype_code")]:
        feats[col] = df[src].astype("category").cat.codes.values
    feats["max_score"]         = df["max_score"].values
    feats["log_max_score"]     = np.log1p(df["max_score"].values)
    feats["score_granularity"] = 100.0 / df["max_score"].values
    feats["ttype_id_code"]     = df["test_type_id"].astype("category").cat.codes.values
    feats["y"]          = df["y"].values
    feats["ts"]         = dt.values
    feats["student_id"] = df["student_id"].values
    feats["discipline"] = df["discipline"].values
    feats["skill_name"] = df["skill_name"].values
    return feats


def build_aggregate_features(feats):
    df = feats.sort_values(["student_id","discipline","ts"]).copy()
    g  = df.groupby(["student_id","discipline"])["y"]
    df["lag1"]        = g.transform(lambda s: s.shift(1))
    df["roll3"]       = g.transform(lambda s: s.shift(1).rolling(3,  min_periods=1).mean())
    df["roll5"]       = g.transform(lambda s: s.shift(1).rolling(5,  min_periods=1).mean())
    df["roll10"]      = g.transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    df["exp_avg"]     = g.transform(lambda s: s.shift(1).expanding().mean())
    df["exp_std"]     = g.transform(lambda s: s.shift(1).expanding().std())
    df["roll5_std"]   = g.transform(lambda s: s.shift(1).rolling(5, min_periods=2).std())
    roll5_first       = g.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean().shift(4))
    df["trend_5v5"]   = df["roll5"] - roll5_first
    df["test_num"]    = df.groupby(["student_id","discipline"]).cumcount()
    gs = df.groupby(["student_id","skill_name"])["y"]
    df["skill_lag1"]    = gs.transform(lambda s: s.shift(1))
    df["skill_roll3"]   = gs.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["skill_exp_avg"] = gs.transform(lambda s: s.shift(1).expanding().mean())
    df["skill_n"]       = gs.transform("cumcount")
    return df.sort_values("ts").reset_index(drop=True)


def add_derived_features(df):
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
    df["e_hat_residual"]   = df["e_hat"] * 100 - df["exp_avg"].fillna(df["e_hat"] * 100)
    df["ability_diff_gap"] = df["theta_skill"] - df["diff_skill"]
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def add_session_features(feats):
    df = feats.sort_values(["student_id","ts"]).copy()
    ts_s    = pd.to_datetime(df["ts"]).astype("int64").values / 1e9
    sid_arr = df["student_id"].values
    new_stu = np.concatenate([[True], sid_arr[1:] != sid_arr[:-1]])
    gap_arr = np.concatenate([[0.0], ts_s[1:] - ts_s[:-1]])
    df["_new_sess"] = (new_stu | (gap_arr > SESSION_GAP_S)).astype(int)
    df["_sess_id"]  = df.groupby("student_id")["_new_sess"].cumsum()
    df["session_pos"] = df.groupby(["student_id","_sess_id"]).cumcount()
    df["session_score_ma"] = df.groupby(["student_id","_sess_id"])["y"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
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
    drop_cols = ["disc_mean_score","disc_std_score",
                 "skill_mean_score","skill_std_score","theta_vs_disc"]
    feats_all = feats_all.drop(columns=[c for c in drop_cols if c in feats_all.columns])
    tr = feats_all[train_mask]
    g_mean, g_std = tr["y"].mean(), tr["y"].std()
    disc_agg  = tr.groupby("discipline")["y"].agg(disc_mean_score="mean",
                                                   disc_std_score="std").reset_index()
    skill_agg = tr.groupby("skill_name")["y"].agg(skill_mean_score="mean",
                                                    skill_std_score="std").reset_index()
    feats_all = feats_all.merge(disc_agg,  on="discipline",  how="left")
    feats_all = feats_all.merge(skill_agg, on="skill_name",  how="left")
    for col, fill in [("disc_mean_score", g_mean),("disc_std_score", g_std),
                      ("skill_mean_score",g_mean),("skill_std_score",g_std)]:
        feats_all[col] = feats_all[col].fillna(fill)
    feats_all["theta_vs_disc"] = feats_all["theta_skill"] - feats_all["disc_mean_score"] / 100.0
    return feats_all


def chrono_split(feats, train_f=0.70, cal_f=0.10):
    n  = len(feats)
    i1 = int(n * train_f)
    i2 = int(n * (train_f + cal_f))
    return feats.iloc[:i1].copy(), feats.iloc[i1:i2].copy(), feats.iloc[i2:].copy()


def temporal_weights(ts_series, halflife=WEIGHT_HALFLIFE):
    dt  = pd.to_datetime(ts_series)
    age = (dt.max() - dt).dt.days.values.astype(float)
    w   = np.exp(-np.log(2.0) * age / halflife)
    return w / w.mean()


def xgb_fit(X_tr, y_tr, X_val=None, y_val=None, sw=None, params=None):
    """Train XGBoost; X_val/y_val ignored (no eval_set — use fixed n_estimators)."""
    p = params if params else XGB_PARAMS
    m = XGBRegressor(**p)
    m.fit(X_tr, np.asarray(y_tr, dtype=np.float64), sample_weight=sw, verbose=False)
    return m


def get_xy(df, feats):
    return df[feats].fillna(0).values, df["y"].values


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATA AUDIT
# ══════════════════════════════════════════════════════════════════════════════

def phase1_data_audit(raw_df, feats_all):
    hdr("P1: DATA AUDIT")
    n_raw  = len(raw_df) + raw_df["skill_name"].isin(PSEUDO).sum()  # approx raw
    n_clean= len(feats_all)
    n_stu  = feats_all["student_id"].nunique()
    n_disc = feats_all["discipline"].nunique()
    n_skill= feats_all["skill_name"].nunique()
    dt     = pd.to_datetime(feats_all["ts"])
    y      = feats_all["y"]

    rec("P1","n_records",    n_clean)
    rec("P1","n_students",   n_stu)
    rec("P1","n_disciplines",n_disc)
    rec("P1","n_skills",     n_skill)
    rec("P1","date_min",     str(dt.min().date()))
    rec("P1","date_max",     str(dt.max().date()))
    rec("P1","score_mean",   round(float(y.mean()), 2))
    rec("P1","score_std",    round(float(y.std()),  2))
    rec("P1","score_median", round(float(y.median()),2))

    log("\n  Records per year:")
    yr_cnt = feats_all.groupby(dt.dt.year).size()
    for yr, cnt in yr_cnt.items():
        log(f"    {yr}: {cnt:>9,}")
        _results.setdefault("P1",{}).setdefault("per_year",{})[str(yr)] = int(cnt)

    log("\n  Top-10 disciplines by volume:")
    disc_cnt = feats_all.groupby("discipline").size().nlargest(10)
    for d, cnt in disc_cnt.items():
        log(f"    {d[:30]:30s}  {cnt:,}")

    log("\n  Tests per student (percentiles):")
    tps = feats_all.groupby("student_id").size()
    for p in [10,25,50,75,90]:
        v = int(np.percentile(tps, p))
        log(f"    P{p:2d}: {v}")
        _results["P1"][f"tps_p{p}"] = v


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — LEAKAGE AUDIT
# ══════════════════════════════════════════════════════════════════════════════

def phase2_leakage_audit(feats_all):
    hdr("P2: LEAKAGE AUDIT — 4 evaluation protocols")
    log("  Protocol A: Leaky features + random split  (inflated)")
    log("  Protocol B: Leaky features + chrono split  (partial fix)")
    log("  Protocol C: GRACE features + random split  (partial fix)")
    log("  Protocol D: GRACE features + chrono split  (honest — our evaluation)")

    # Leaky: include lag1 computed without shift (i.e. current score visible)
    fa = feats_all.copy()
    fa["lag0_leaky"] = fa["y"]   # direct leakage: current score as feature
    leaky_feats = CORE + ["lag0_leaky","lag1","roll3","roll5"]

    FAST_P = dict(n_estimators=300, learning_rate=0.08, max_depth=6,
                  tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0)

    def _fit_eval(X_tr, y_tr, X_te, y_te):
        m = XGBRegressor(**FAST_P)
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        return r2_score(y_te, p), mean_absolute_error(y_te, p)

    X_leaky = fa[leaky_feats].fillna(0).values
    X_grace = fa[ALL_FEATS].fillna(0).values
    y_all   = fa["y"].values
    n       = len(fa)

    # Random 80/20 split
    rng  = np.random.default_rng(SEED)
    idx  = rng.permutation(n)
    tr_r = idx[:int(0.8*n)]
    te_r = idx[int(0.8*n):]

    # Chrono 70/20 split
    tr_c = np.arange(int(0.7*n))
    te_c = np.arange(int(0.8*n), n)

    for label, proto, X, tr_idx, te_idx in [
        ("A — Leaky+Random",  "A", X_leaky, tr_r, te_r),
        ("B — Leaky+Chrono",  "B", X_leaky, tr_c, te_c),
        ("C — GRACE+Random",  "C", X_grace, tr_r, te_r),
        ("D — GRACE+Chrono",  "D", X_grace, tr_c, te_c),
    ]:
        r2, mae = _fit_eval(X[tr_idx], y_all[tr_idx], X[te_idx], y_all[te_idx])
        log(f"  {label:25s}  R²={r2:.3f}  MAE={mae:.2f} pp")
        _results.setdefault("P2",{})[f"proto_{proto}"] = {"r2": round(r2,4),
                                                           "mae": round(mae,3)}

    log("\n  Leakage inflation = Protocol A − Protocol D")
    inf = _results["P2"]["proto_A"]["r2"] - _results["P2"]["proto_D"]["r2"]
    rec("P2","leakage_inflation_r2", round(inf, 4))


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — BASELINE MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def phase3_baselines(train, cal, test):
    hdr("P3: BASELINE MODEL COMPARISON (same features/split)")
    feats = ALL_FEATS
    X_tr, y_tr = get_xy(train, feats)
    X_ca, y_ca = get_xy(cal,   feats)
    X_te, y_te = get_xy(test,  feats)

    # Combine train+cal for non-early-stop models
    X_trca = np.vstack([X_tr, X_ca])
    y_trca = np.concatenate([y_tr, y_ca])

    results_p3 = {}

    # Ridge
    t0 = time.time()
    sc = StandardScaler()
    m  = Ridge(alpha=1.0)
    m.fit(sc.fit_transform(X_trca), y_trca)
    p  = m.predict(sc.transform(X_te))
    r2, mae = r2_score(y_te, p), mean_absolute_error(y_te, p)
    log(f"  Ridge      R²={r2:.4f}  MAE={mae:.3f}  [{time.time()-t0:.0f}s]")
    results_p3["Ridge"] = {"r2": round(r2,4), "mae": round(mae,3)}

    # Random Forest (fast: 200 trees, max_depth=12)
    t0 = time.time()
    m  = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=20,
                                n_jobs=-1, random_state=SEED)
    m.fit(X_trca, y_trca)
    p  = m.predict(X_te)
    r2, mae = r2_score(y_te, p), mean_absolute_error(y_te, p)
    log(f"  RF(200)    R²={r2:.4f}  MAE={mae:.3f}  [{time.time()-t0:.0f}s]")
    results_p3["RandomForest"] = {"r2": round(r2,4), "mae": round(mae,3)}

    # XGBoost (fast, no early stopping)
    t0 = time.time()
    m  = XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=8,
                       subsample=0.8, colsample_bytree=0.7,
                       tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0)
    m.fit(X_trca, y_trca)
    p  = m.predict(X_te)
    r2, mae = r2_score(y_te, p), mean_absolute_error(y_te, p)
    log(f"  XGBoost    R²={r2:.4f}  MAE={mae:.3f}  [{time.time()-t0:.0f}s]")
    results_p3["XGBoost"] = {"r2": round(r2,4), "mae": round(mae,3)}

    _results["P3"] = results_p3


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — GRACE ACCURACY (main model)
# ══════════════════════════════════════════════════════════════════════════════

def phase4_grace_accuracy(model, train, cal, test):
    hdr("P4: GRACE POINT ACCURACY")
    feats = ALL_FEATS
    X_tr, y_tr = get_xy(train, feats)
    X_ca, y_ca = get_xy(cal,   feats)
    X_te, y_te = get_xy(test,  feats)

    p_tr = model.predict(X_tr)
    p_te = model.predict(X_te)

    r2_tr  = r2_score(y_tr, p_tr)
    r2_te  = r2_score(y_te, p_te)
    mae_te = mean_absolute_error(y_te, p_te)
    rmse   = float(np.sqrt(np.mean((y_te - p_te)**2)))
    gap    = r2_tr - r2_te

    rec("P4","r2_train", round(r2_tr, 4))
    rec("P4","r2_test",  round(r2_te, 4))
    rec("P4","mae_test", round(mae_te, 3))
    rec("P4","rmse_test",round(rmse, 3))
    rec("P4","gen_gap",  round(gap, 4))

    # Per-student median R² (students with ≥8 test rows)
    te = test.copy()
    te["pred"] = p_te
    def _s_r2(g):
        if len(g) < 8:
            return np.nan
        return r2_score(g["y"], g["pred"])
    sr2 = te.groupby("student_id").apply(_s_r2).dropna()
    rec("P4","per_student_r2_median", round(float(sr2.median()), 4))
    rec("P4","per_student_r2_mean",   round(float(sr2.mean()),   4))
    rec("P4","per_student_n",         int(len(sr2)))
    log(f"  Per-student R² (n={len(sr2):,}): "
        f"median={sr2.median():.4f}  mean={sr2.mean():.4f}  "
        f"P25={sr2.quantile(0.25):.4f}  P75={sr2.quantile(0.75):.4f}")

    return p_te


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — ROLLING-ORIGIN VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def phase5_rolling_origin(feats_all):
    hdr("P5: ROLLING-ORIGIN TEMPORAL VALIDATION (5 folds)")
    dt   = pd.to_datetime(feats_all["ts"])
    yrs  = [2021, 2022, 2023, 2024, 2025]
    r2s_grace, r2s_base = [], []

    # Use fast XGB params to keep runtime tractable per fold
    FAST_P = dict(n_estimators=500, learning_rate=0.05, max_depth=8,
                  subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                  reg_lambda=2.0, reg_alpha=0.2, objective="reg:squarederror",
                  tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0)

    for yr in yrs:
        tr_mask  = dt.dt.year < yr
        te_mask  = dt.dt.year == yr
        if te_mask.sum() < 100:
            log(f"  {yr}: skipped (n_test={te_mask.sum()})")
            continue

        tr = feats_all[tr_mask].copy()
        te = feats_all[te_mask].copy()

        # Add static content features from training fold only
        all_mask = np.zeros(len(feats_all), dtype=bool)
        all_mask[tr_mask] = True
        fold = add_static_content_features(feats_all.copy(), all_mask)
        tr_f = fold[tr_mask]
        te_f = fold[te_mask]

        X_tr, y_tr = get_xy(tr_f, ALL_FEATS)
        X_te, y_te = get_xy(te_f, ALL_FEATS)
        X_tr_c = tr_f[CORE].fillna(0).values
        X_te_c = te_f[CORE].fillna(0).values
        y_tr_c = tr_f["y"].values

        # GRACE model
        m_g = XGBRegressor(**FAST_P)
        m_g.fit(X_tr, y_tr)
        r2_g = r2_score(y_te, m_g.predict(X_te))

        # Baseline (CORE features only)
        m_b = XGBRegressor(**FAST_P)
        m_b.fit(X_tr_c, y_tr_c)
        r2_b = r2_score(y_te, m_b.predict(X_te_c))

        r2s_grace.append(r2_g)
        r2s_base.append(r2_b)
        log(f"  {yr}: GRACE={r2_g:.4f}  Baseline={r2_b:.4f}  "
            f"delta={r2_g-r2_b:+.4f}  (n_test={te_mask.sum():,})")
        _results.setdefault("P5",{}).setdefault("folds",{})[str(yr)] = {
            "r2_grace": round(r2_g, 4), "r2_base": round(r2_b, 4),
            "n_test": int(te_mask.sum())
        }

    if r2s_grace:
        rec("P5","mean_r2",  round(float(np.mean(r2s_grace)), 4))
        rec("P5","std_r2",   round(float(np.std(r2s_grace)),  4))
        rec("P5","mean_r2_base", round(float(np.mean(r2s_base)), 4))
        log(f"\n  GRACE rolling: {np.mean(r2s_grace):.4f} ± {np.std(r2s_grace):.4f}")
        log(f"  Base  rolling: {np.mean(r2s_base):.4f} ± {np.std(r2s_base):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — HISTORY DEPTH
# ══════════════════════════════════════════════════════════════════════════════

def phase6_history_depth(model, test, p_te):
    hdr("P6: HISTORY DEPTH STRATIFICATION")
    te = test.copy()
    te["pred"] = p_te
    bins   = [0, 4, 10, 20, 9999]
    labels = ["1–4","5–10","11–20",">20"]
    te["depth"] = pd.cut(te["n_disc"], bins=bins, labels=labels)
    log(f"  {'Depth':8s}  {'N':>9s}  {'R²':>7s}  {'MAE':>7s}")
    res_p6 = {}
    for lbl in labels:
        g = te[te["depth"]==lbl]
        if len(g) < 50:
            continue
        r2  = r2_score(g["y"], g["pred"])
        mae = mean_absolute_error(g["y"], g["pred"])
        log(f"  {lbl:8s}  {len(g):>9,}  {r2:>7.4f}  {mae:>7.3f}")
        res_p6[lbl] = {"n": int(len(g)), "r2": round(r2,4), "mae": round(mae,3)}
    _results["P6"] = res_p6


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — FEATURE GROUP ABLATION
# ══════════════════════════════════════════════════════════════════════════════

def phase7_ablation(train, cal, test):
    hdr("P7: FEATURE GROUP ABLATION")
    X_ca, y_ca = get_xy(cal, ALL_FEATS)

    configs = [
        ("GRACE (Core+A+B+C — 70 f.)", ALL_FEATS),
        ("Without Group C  (60 f.)",    CORE + GROUP_A + GROUP_B),
        ("Without B+C      (46 f.)",    CORE + GROUP_A),
        ("Core only        (26 f.)",    CORE),
    ]
    log(f"  {'Config':30s}  {'N feats':>7s}  {'R²':>7s}  {'MAE':>7s}  {'ΔR²':>8s}")
    base_r2 = None
    res_p7  = {}
    for label, feats in configs:
        X_tr, y_tr = get_xy(train, feats)
        _X_ca = cal[feats].fillna(0).values
        X_te, y_te = get_xy(test, feats)
        sw = temporal_weights(train["ts"])
        m  = xgb_fit(X_tr, y_tr, _X_ca, y_ca, sw=sw, params=XGB_FAST)
        p  = m.predict(X_te)
        r2  = r2_score(y_te, p)
        mae = mean_absolute_error(y_te, p)
        delta = (r2 - base_r2) if base_r2 is not None else 0.0
        if base_r2 is None:
            base_r2 = r2
        log(f"  {label:30s}  {len(feats):>7d}  {r2:>7.4f}  {mae:>7.3f}  {delta:>+8.4f}")
        res_p7[label] = {"n_feats": len(feats), "r2": round(r2,4),
                         "mae": round(mae,3), "delta_r2": round(delta,4)}
    _results["P7"] = res_p7


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 8 — CONFORMAL INTERVALS
# ══════════════════════════════════════════════════════════════════════════════

def phase8_conformal(model, cal, test):
    hdr("P8: CONFORMAL PREDICTION INTERVALS")
    X_ca, y_ca = get_xy(cal, ALL_FEATS)
    X_te, y_te = get_xy(test, ALL_FEATS)
    p_ca = model.predict(X_ca)
    p_te = model.predict(X_te)
    scores_ca = np.abs(y_ca - p_ca)   # residual conformity scores

    def _coverage_width(lo, hi, y):
        cov   = float(np.mean((y >= lo) & (y <= hi)))
        width = float(np.mean(hi - lo))
        return cov, width

    # ── Mondrian CQR (stratify by rd_disc tertile) ────────────────────────────
    rd_ca = cal["rd_disc"].fillna(cal["rd_disc"].mean()).values
    rd_te = test["rd_disc"].fillna(cal["rd_disc"].mean()).values
    t33, t67 = np.percentile(rd_ca, [33, 67])

    def _stratum(rd, t33, t67):
        return np.where(rd < t33, 0, np.where(rd < t67, 1, 2))

    st_ca = _stratum(rd_ca, t33, t67)
    st_te = _stratum(rd_te, t33, t67)
    strata_lbls = ["Low RD", "Mid RD", "High RD"]

    mon_lo = np.full(len(X_te), np.nan)
    mon_hi = np.full(len(X_te), np.nan)
    mon_strata = {}
    for s in range(3):
        m_ca = st_ca == s
        m_te = st_te == s
        if m_ca.sum() < 10:
            continue
        q   = np.quantile(scores_ca[m_ca], 1 - ALPHA)
        mon_lo[m_te] = p_te[m_te] - q
        mon_hi[m_te] = p_te[m_te] + q
        cov_s, wid_s = _coverage_width(
            mon_lo[m_te], mon_hi[m_te], y_te[m_te]
        )
        n_s = int(m_te.sum())
        log(f"  Mondrian [{strata_lbls[s]:8s}]: n={n_s:>7,}  "
            f"cov={cov_s:.4f}  width={wid_s:.2f} pp")
        mon_strata[strata_lbls[s]] = {"n": n_s, "coverage": round(cov_s,4),
                                       "width": round(wid_s,2)}

    valid = ~np.isnan(mon_lo)
    cov_mon, wid_mon = _coverage_width(mon_lo[valid], mon_hi[valid], y_te[valid])
    log(f"  Mondrian  overall: cov={cov_mon:.4f}  width={wid_mon:.2f} pp")

    # ── Asymmetric CQR (by predicted-score band) ─────────────────────────────
    bands_ca = pd.cut(p_ca, bins=[-np.inf,30,50,70,90,np.inf], labels=False)
    bands_te = pd.cut(p_te, bins=[-np.inf,30,50,70,90,np.inf], labels=False)
    sigma_ca = np.ones(len(p_ca))
    sigma_map = {}
    for b in range(5):
        m_b = bands_ca == b
        if m_b.sum() >= 10:
            sigma_map[b] = max(np.std(scores_ca[m_b]), 1.0)
        else:
            sigma_map[b] = max(np.std(scores_ca), 1.0)
    for b in range(5):
        sigma_ca[bands_ca == b] = sigma_map.get(b, sigma_map.get(0, 1.0))

    norm_scores = scores_ca / sigma_ca
    q_va = np.quantile(norm_scores, 1 - ALPHA)

    sigma_te = np.array([sigma_map.get(int(b) if not np.isnan(b) else 2, 1.0)
                         for b in bands_te])
    va_lo = p_te - q_va * sigma_te
    va_hi = p_te + q_va * sigma_te
    cov_va, wid_va = _coverage_width(va_lo, va_hi, y_te)
    log(f"  Asym.CQR  overall: cov={cov_va:.4f}  width={wid_va:.2f} pp")

    _results["P8"] = {
        "mondrian": {"coverage": round(cov_mon,4), "width": round(wid_mon,2),
                     "strata": mon_strata},
        "va_cqr":   {"coverage": round(cov_va,4),  "width": round(wid_va,2)},
        "target_coverage": 1 - ALPHA,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 9 — PER-DISCIPLINE
# ══════════════════════════════════════════════════════════════════════════════

def phase9_per_discipline(model_grace, model_base, train, cal, test):
    hdr("P9: PER-DISCIPLINE R²")
    feats_g = ALL_FEATS
    feats_b = CORE

    te = test.copy()
    te["pred_grace"] = model_grace.predict(te[feats_g].fillna(0).values)
    te["pred_base"]  = model_base.predict( te[feats_b].fillna(0).values)

    discs = te.groupby("discipline").size()
    discs = discs[discs >= 100].index.tolist()

    log(f"\n  {'Discipline':25s}  {'N':>8s}  {'R²_base':>8s}  {'R²_GRACE':>8s}  {'Δ':>8s}")
    res_p9 = []
    for d in sorted(discs):
        g = te[te["discipline"] == d]
        r2_b = r2_score(g["y"], g["pred_base"])
        r2_g = r2_score(g["y"], g["pred_grace"])
        delta = r2_g - r2_b
        log(f"  {d[:25]:25s}  {len(g):>8,}  {r2_b:>8.4f}  {r2_g:>8.4f}  {delta:>+8.4f}")
        res_p9.append({"discipline": d, "n": int(len(g)),
                       "r2_base": round(r2_b,4), "r2_grace": round(r2_g,4),
                       "delta": round(delta,4)})
    _results["P9"] = sorted(res_p9, key=lambda x: -x["r2_grace"])


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 10 — BOOTSTRAP SIGNIFICANCE
# ══════════════════════════════════════════════════════════════════════════════

def phase10_bootstrap(model_grace, model_base, test, B=2000):
    hdr("P10: BOOTSTRAP SIGNIFICANCE TEST")
    log(f"  Comparing GRACE vs Baseline (Core features) — B={B} bootstrap samples")

    feats_g = ALL_FEATS
    feats_b = CORE
    X_te_g  = test[feats_g].fillna(0).values
    X_te_b  = test[feats_b].fillna(0).values
    y_te    = test["y"].values
    p_g     = model_grace.predict(X_te_g)
    p_b     = model_base.predict(X_te_b)

    n = len(y_te)
    rng = np.random.default_rng(SEED)
    boot_delta = np.empty(B)
    for i in range(B):
        idx       = rng.integers(0, n, n)
        boot_delta[i] = r2_score(y_te[idx], p_g[idx]) - r2_score(y_te[idx], p_b[idx])

    observed  = r2_score(y_te, p_g) - r2_score(y_te, p_b)
    p_value   = float(np.mean(boot_delta <= 0))
    ci_lo     = float(np.percentile(boot_delta, 2.5))
    ci_hi     = float(np.percentile(boot_delta, 97.5))

    rec("P10","observed_delta_r2",   round(observed, 4))
    rec("P10","boot_mean_delta",     round(float(np.mean(boot_delta)), 4))
    rec("P10","ci95_lo",             round(ci_lo, 4))
    rec("P10","ci95_hi",             round(ci_hi, 4))
    rec("P10","p_value",             round(p_value, 6))
    rec("P10","significant_0.05",    p_value < 0.05)
    log(f"\n  ΔR² observed = {observed:+.4f}")
    log(f"  95% CI       = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    log(f"  p-value      = {p_value:.6f}  {'✓ significant' if p_value < 0.05 else '✗ not significant'}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="5000-student subsample for quick testing")
    args = parser.parse_args()

    t_start = time.time()
    fast_n  = 5000 if args.fast else None

    log("GRACE — 10-Phase Comprehensive Evaluation")
    log(f"Mode: {'FAST (5k students)' if args.fast else 'FULL'}")
    log(f"Started: {pd.Timestamp.now()}")

    # ── Data loading ──────────────────────────────────────────────────────────
    raw_df = load_and_clean(fast_n=fast_n)

    # ── Feature engineering ───────────────────────────────────────────────────
    hdr("FEATURE ENGINEERING")
    t0 = time.time()
    feats = build_rating_features(raw_df)
    feats = build_aggregate_features(feats)
    feats = add_derived_features(feats)
    feats = add_session_features(feats)
    feats = add_exam_features(feats)
    log(f"  Feature engineering: {time.time()-t0:.1f}s")

    # Chronological split
    train, cal, test = chrono_split(feats)
    train_mask = np.zeros(len(feats), dtype=bool)
    train_mask[:len(train)] = True
    feats = add_static_content_features(feats, train_mask)

    # Re-slice after static features were added
    n1, n2 = len(train), len(train) + len(cal)
    train = feats.iloc[:n1].copy()
    cal   = feats.iloc[n1:n2].copy()
    test  = feats.iloc[n2:].copy()
    log(f"  Split — train: {len(train):,}  cal: {len(cal):,}  test: {len(test):,}")

    # ── Train main GRACE model ────────────────────────────────────────────────
    hdr("TRAINING MAIN GRACE MODEL (XGBoost + temporal weights)")
    t0 = time.time()
    params = XGB_FAST if args.fast else XGB_PARAMS
    X_tr, y_tr = get_xy(train, ALL_FEATS)
    X_ca, y_ca = get_xy(cal,   ALL_FEATS)
    sw = temporal_weights(train["ts"])
    model_grace = xgb_fit(X_tr, y_tr, X_ca, y_ca, sw=sw, params=params)
    log(f"  GRACE model trained in {time.time()-t0:.1f}s")

    # Baseline model (CORE features only, same XGB setup)
    hdr("TRAINING BASELINE MODEL (Core 26 features)")
    t0 = time.time()
    X_tr_b = train[CORE].fillna(0).values
    X_ca_b = cal[CORE].fillna(0).values
    model_base = xgb_fit(X_tr_b, y_tr, X_ca_b, y_ca, sw=sw, params=params)
    log(f"  Baseline trained in {time.time()-t0:.1f}s")

    # ── Run all 10 phases ─────────────────────────────────────────────────────
    phase1_data_audit(raw_df, feats)
    flush_outputs()

    phase2_leakage_audit(feats)
    flush_outputs()

    phase3_baselines(train, cal, test)
    flush_outputs()

    p_te = phase4_grace_accuracy(model_grace, train, cal, test)
    flush_outputs()

    phase5_rolling_origin(feats)
    flush_outputs()

    phase6_history_depth(model_grace, test, p_te)
    flush_outputs()

    phase7_ablation(train, cal, test)
    flush_outputs()

    phase8_conformal(model_grace, cal, test)
    flush_outputs()

    phase9_per_discipline(model_grace, model_base, train, cal, test)
    flush_outputs()

    phase10_bootstrap(model_grace, model_base, test,
                      B=500 if args.fast else 2000)
    flush_outputs()

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = (time.time() - t_start) / 60
    hdr("EVALUATION COMPLETE")
    log(f"  Total runtime: {elapsed:.1f} min")
    log(f"  Results → {RES_PATH}")
    log(f"  Log     → {LOG_PATH}")

    p4 = _results.get("P4", {})
    p10= _results.get("P10",{})
    log(f"\n  GRACE R²={p4.get('r2_test','?')}  MAE={p4.get('mae_test','?')} pp")
    log(f"  ΔR² vs baseline={p10.get('observed_delta_r2','?')}  "
        f"p={p10.get('p_value','?')}")
    flush_outputs()


if __name__ == "__main__":
    main()
