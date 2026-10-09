"""
run_grace_v7.py — GRACE V7: quantile widening + 3 new features + drop 2 weak features

78 features: V6 base (77) - 2 dropped + 3 new
Expected: R²~0.558-0.562, coverage >=0.88, width ~45-48 pp

Changes over V6 (R²=0.556, coverage=0.859, width=44.4):
  FIX Quantile heads widened: alpha 0.05/0.95 -> 0.03/0.97
        V6 Q_lo=Q_hi=0.000 because model already ~86% covered; conformal did nothing.
        Wider heads give raw ~94% nominal coverage, conformal correction then trims
        back to exact 90%, yielding Q_lo>0, Q_hi>0 and valid coverage >=0.88-0.90.
        Constant QUANTILE_ALPHA = 0.06 (heads at 3rd/97th pct).

  #O  3 new features (78 total):
        best_disc_score = running max score in discipline (ceiling signal, no lookahead)
        theta_disc_sq   = theta_disc^2 (non-linear ability: quadratic effect near ceiling)
        gap_trend       = days_gap_disc - days_gap_skill (skill gap wider than disc = rusty)

  DROP 2 weakest V6/V4 features:
        attempt_density_30d  (rank #67, 0.6% — weak, correlated with n_disc)
        in_exam_season       (rank #74, 0.1% — nearly zero signal)

Usage:  python run_grace_v7.py
Output: results/run_log_v7.txt, results/results_v7.json
"""

import sys, io, time, json, warnings
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
SEED = 42
ALPHA = 0.10
QUANTILE_ALPHA = 0.06   # V7: quantile heads at 3rd/97th pct (wider than V6's 5th/95th)
np.random.seed(SEED)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ── Glicko constants (same as v3) ─────────────────────────────────────────────
K0, V_GAIN, RD0, RD_MIN, C_DECAY, K_ITEM = 0.35, 0.04, 0.35, 0.06, 0.010, 0.02
EMA_HALFLIFE = 90.0   # days

# Session gap threshold (seconds): >1 hour = new study session
SESSION_GAP_S = 3600.0

# Temporal weight half-life (days): data 2 years old gets weight 0.5
WEIGHT_HALFLIFE = 730.0

PSEUDO_SKILL_NAMES = [
    'Номер теста','Длительность','Темы','Темы среза/дз (прошлый урок)',
    'Количественные характеристики','Ментор','Туториал','Время потока',
    'Оценка учителя','Школа','Отдел Менторов','Обратная связь получена',
    'КРА','МКЕ','Оценка среза','Баллы за тест','Попытки','Ранг',
    'Комментарий','Комментарий.','Срез Общее',
]

OUT = Path("results")
OUT.mkdir(exist_ok=True)
LOG_PATH = OUT / "run_log_v7.txt"
RES_PATH = OUT / "results_v7.json"

# ── Forgetting curve time constants (Strategy C) ───────────────────────────────
TAU_DISC  = 30.0   # days: per-discipline retention half-life
TAU_SKILL = 20.0   # days: per-skill retention half-life (faster decay)
MIN_STRAT_N = 50   # min calibration samples per Mondrian stratum (Strategy A)

# ── Feature lists ──────────────────────────────────────────────────────────────
GRACE_V2_FEATURES = [
    "theta_g","theta_disc","theta_skill","rd_g","rd_disc",
    "diff_skill","diff_ttype","diff_skill_ema","e_hat","surprise5",
    "days_gap","n_global","n_disc","n_skill",
    "days_gap_disc","days_gap_skill",
    "month","dayofweek","disc_code","skill_code","ttype_code",
    "max_score","log_max_score","score_granularity","ttype_id_code",
    "lag1","roll3","roll5","roll10","exp_avg","exp_std",
    "skill_lag1","skill_roll3","skill_exp_avg","skill_n",
    "test_num","trend_5v5","roll5_std",
    "theta_range","score_momentum","gap_bucket","gap_bucket_disc",
    "is_cold","is_cold_skill","e_hat_residual","ability_diff_gap",
]  # 46 features

GRACE_V3_EXTRA = [
    "rd_skill",                                              # per-skill RD (v3 fix)
    "student_tenure","disc_breadth","skill_breadth",         # platform context
    "intensity",                                             # tests per day
    "disc_mean_score","disc_std_score",                      # static disc difficulty
    "skill_mean_score","skill_std_score",                    # static skill difficulty
    "theta_vs_disc",                                         # relative ability
    "month_sin","month_cos","dow_sin","dow_cos",              # cyclical time
]  # 14 features

GRACE_V4_EXTRA = [
    # #F: global cross-discipline rolling stats
    "global_lag1","global_roll3","global_roll5","global_roll10",
    "global_std5","ability_gap_global",
    # #G: session context
    "session_pos","session_score_ma",
    # #H: UNT exam proximity
    "days_to_june","in_exam_season",
]  # 10 features

GRACE_V4_FEATURES = GRACE_V2_FEATURES + GRACE_V3_EXTRA + GRACE_V4_EXTRA  # 70 total

GRACE_V5_EXTRA = [
    # #K: forgetting curve (Strategy C)
    "forget_disc",     # exp(-days_gap_disc / TAU_DISC)
    "forget_skill",    # exp(-days_gap_skill / TAU_SKILL)
    "expect_retain",   # theta_disc * forget_disc  (ability-weighted retention signal)
]  # 3 features

GRACE_V5_FEATURES = GRACE_V4_FEATURES + GRACE_V5_EXTRA  # 73 total

GRACE_V6_EXTRA = [
    # #N: score velocity (V6 new)
    "score_slope_disc",     # slope of last 5 per-discipline pct scores (pp per attempt)
    "score_slope_global",   # slope of last 5 global pct scores
    "attempt_density_30d",  # attempts in last 30 days (study intensity proxy)
    "rd_skill_gap",         # rd_skill - rd_disc (skill uncertainty above disc level)
]  # 4 features

GRACE_V6_FEATURES = GRACE_V5_FEATURES + GRACE_V6_EXTRA  # 77 total

GRACE_V7_DROP = {"attempt_density_30d", "in_exam_season"}  # rank #67/0.6% and #74/0.1%

GRACE_V7_EXTRA = [
    # #O: new V7 features
    "best_disc_score",  # running max score in discipline (ceiling signal, no lookahead)
    "theta_disc_sq",    # theta_disc^2 (non-linear ability near ceiling)
    "gap_trend",        # days_gap_disc - days_gap_skill (skill gap wider than disc = rusty)
]  # 3 features

GRACE_V7_FEATURES = (
    [f for f in GRACE_V6_FEATURES if f not in GRACE_V7_DROP] + GRACE_V7_EXTRA
)  # 77 - 2 + 3 = 78 total

XGB_BASE = dict(
    n_estimators=2000, learning_rate=0.04, max_depth=8,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
    reg_lambda=2.0, reg_alpha=0.2, objective="reg:squarederror",
    tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0,
    early_stopping_rounds=50,
)

LGB_BASE = dict(
    n_estimators=5000, learning_rate=0.02, num_leaves=255,
    subsample=0.8, subsample_freq=5, colsample_bytree=0.7,
    min_child_samples=50, reg_alpha=0.1, reg_lambda=1.5,
    n_jobs=-1, random_state=SEED, verbose=-1,
)

# ── Logging helpers ────────────────────────────────────────────────────────────
_log_lines = []
_results   = {}

def log(msg=""):
    print(msg, flush=True)
    _log_lines.append(str(msg))

def rec(key, val):
    _results[key] = str(val)
    log(f"  {key} = {val}")

def hdr(title):
    log("\n" + "="*70)
    log(title)
    log("="*70)


# ==============================================================================
# STEP 1 — DATA LOADING
# ==============================================================================
def load_and_clean():
    hdr("STEP 1: DATA LOADING & CLEANING")
    df = pd.read_csv("report/student_performance_data.csv", low_memory=False)
    raw_n   = len(df)
    raw_stu = df["student_id"].nunique() if "student_id" in df.columns else 0
    log(f"  Raw rows before any cleaning: {raw_n:,}  unique students: {raw_stu:,}")
    rec("DATA_RAW_NRECORDS",  f"{raw_n:,}")
    rec("DATA_RAW_NSTUDENTS", f"{raw_stu:,}")
    df = df[~df["skill_name"].isin(PSEUDO_SKILL_NAMES)].copy()
    df["test_datetime"] = pd.to_datetime(df["test_datetime"], errors="coerce")
    df = df.dropna(subset=["test_datetime"])
    df = df[df["test_datetime"] >= "2018-01-01"]
    df["score"]     = pd.to_numeric(df["score"],     errors="coerce")
    df["max_score"] = pd.to_numeric(df["max_score"], errors="coerce")
    df = df.dropna(subset=["score","max_score"])
    df = df[(df["max_score"] > 2) & (df["score"] >= 0) & (df["score"] <= df["max_score"])]
    df = df.drop_duplicates()
    df["y"] = 100.0 * df["score"] / df["max_score"]
    rec("DATA_NRECORDS",  f"{len(df):,}")
    rec("DATA_NSTUDENTS", f"{df['student_id'].nunique():,}")
    log(f"  Clean corpus: {len(df):,} rows, {df['student_id'].nunique():,} students")
    return df


# ==============================================================================
# STEP 2 — RATING FEATURES V4
# ==============================================================================
def build_rating_features_v4(df, verbose=True):
    """
    V4 extends v3 by adding global cross-discipline rolling statistics (#F).
    Key v3 fixes retained: rd_skill with k_s gain, student context features (#B).
    New: global_lag1/roll3/roll5/roll10/std5, ability_gap_global.
    Output: 27 columns (21 from v3 + 6 global rolling).
    """
    t0 = time.time()
    df = df.sort_values("test_datetime", kind="mergesort").reset_index(drop=True)

    y       = (df["score"] / df["max_score"]).clip(0, 1).to_numpy()
    ts      = pd.to_datetime(df["test_datetime"]).astype("int64").to_numpy() / 86.4e12
    students= df["student_id"].to_numpy()
    discs   = df["discipline"].astype(str).to_numpy()
    skills  = df["skill_name"].astype(str).to_numpy()
    ttypes  = df["test_type_category_id"].astype(str).to_numpy()

    theta_g, theta_d, theta_s = {}, {}, {}
    rd_g_v, rd_d_v, rd_s_v   = {}, {}, {}
    diff_skill, diff_ttype    = {}, {}
    diff_skill_ema            = {}
    last_g, last_d, last_s    = {}, {}, {}
    n_g, n_d, n_s             = {}, {}, {}
    surprise                  = {}
    first_seen                = {}
    disc_sets                 = {}
    skill_sets                = {}
    global_hist               = {}   # #F: per-student deque of recent y values (max 10)

    n   = len(df)
    # 27 output columns: 21 from v3 + 6 global rolling
    out = np.zeros((n, 27), dtype=np.float64)

    EMA_DECAY = np.log(2) / EMA_HALFLIFE

    for t in range(n):
        if verbose and t % 500_000 == 0 and t > 0:
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

        decay_ema = np.exp(-gap_s * EMA_DECAY)
        d_ema     = diff_skill_ema.get(j, 0.0) * decay_ema

        e_hat = 1.0 / (1.0 + np.exp(-(0.5*th_d + 0.5*th_s - 0.7*(dj + d_ema) - 0.3*dc)))
        sur   = surprise.get(s_key, [])

        # #B: student context
        first_t   = first_seen.get(i, ts[t])
        tenure    = ts[t] - first_t
        db        = len(disc_sets.get(i,  set()))
        sb        = len(skill_sets.get(i, set()))
        n_g_val   = n_g.get(i, 0)
        intensity = n_g_val / max(tenure, 1.0)

        # #F: global cross-discipline rolling stats
        hist   = global_hist.get(i, deque(maxlen=10))
        h_list = list(hist)
        if h_list:
            g_lag1  = h_list[-1] * 100.0
            g_roll3 = np.mean(h_list[-3:]) * 100.0
            g_roll5 = np.mean(h_list[-5:]) * 100.0
            g_roll10= np.mean(h_list)       * 100.0
            g_std5  = (np.std(h_list[-5:])  * 100.0 if len(h_list) >= 2 else 0.0)
            # ability_gap_global: expected score vs actual recent global performance
            # e_hat in [0,1], g_roll5 in [0,100] -> convert e_hat to same scale
            g_gap   = e_hat * 100.0 - g_roll5
        else:
            # No history: use e_hat as best prior
            g_lag1  = e_hat * 100.0
            g_roll3 = e_hat * 100.0
            g_roll5 = e_hat * 100.0
            g_roll10= e_hat * 100.0
            g_std5  = 0.0
            g_gap   = 0.0

        out[t] = [
            th_g, th_d, th_s,
            np.sqrt(v_g), np.sqrt(v_d), np.sqrt(v_s),
            dj, dc, d_ema, e_hat,
            np.mean(sur[-5:]) if sur else 0.0,
            min(gap_g, 365.0), min(gap_d, 365.0), min(gap_s, 365.0),
            float(n_g_val), float(n_d.get(s_key, 0)), float(n_s.get(sk, 0)),
            min(tenure, 3650.0),
            float(db), float(sb),
            min(intensity, 20.0),
            # #F: global rolling
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

        last_g[i]      = ts[t]
        last_d[s_key]  = ts[t]
        last_s[sk]     = ts[t]
        n_g[i]         = n_g_val + 1
        n_d[s_key]     = n_d.get(s_key, 0) + 1
        n_s[sk]        = n_s.get(sk,    0) + 1
        surprise[s_key]= (sur + [err])[-5:]

        if i not in first_seen:
            first_seen[i] = ts[t]
        disc_sets.setdefault(i,  set()).add(discs[t])
        skill_sets.setdefault(i, set()).add(skills[t])

        # Update global history AFTER recording output (no lookahead)
        hist.append(y[t])
        global_hist[i] = hist

    elapsed = time.time() - t0
    log(f"    Rating loop v4: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    cols = [
        "theta_g","theta_disc","theta_skill",
        "rd_g","rd_disc","rd_skill",
        "diff_skill","diff_ttype","diff_skill_ema","e_hat","surprise5",
        "days_gap","days_gap_disc","days_gap_skill",
        "n_global","n_disc","n_skill",
        "student_tenure","disc_breadth","skill_breadth","intensity",
        # #F: global rolling
        "global_lag1","global_roll3","global_roll5","global_roll10",
        "global_std5","ability_gap_global",
    ]
    feats = pd.DataFrame(out, columns=cols)

    dt = pd.to_datetime(df["test_datetime"])
    feats["month"]     = dt.dt.month.values
    feats["dayofweek"] = dt.dt.dayofweek.values
    for src, col in [("discipline","disc_code"), ("skill_name","skill_code"),
                     ("test_type_category_id","ttype_code")]:
        feats[col] = df[src].astype("category").cat.codes.values

    feats["max_score"]         = df["max_score"].values
    feats["log_max_score"]     = np.log1p(df["max_score"].values)
    feats["score_granularity"] = 100.0 / df["max_score"].values
    feats["ttype_id_code"]     = df["test_type_id"].astype("category").cat.codes.values

    feats["y"]          = y * 100.0
    feats["ts"]         = dt.values
    feats["student_id"] = df["student_id"].values
    feats["discipline"] = df["discipline"].values
    feats["skill_name"] = df["skill_name"].values

    log(f"  V4 rating matrix: {len(feats):,} rows x {len(cols)+10} columns")
    return feats


# ==============================================================================
# STEP 3a — AGGREGATE + DERIVED FEATURES (same as v3)
# ==============================================================================
def build_aggregate_features(feats):
    df = feats.sort_values(["student_id","discipline","ts"]).copy()
    g  = df.groupby(["student_id","discipline"])["y"]
    df["lag1"]       = g.transform(lambda s: s.shift(1))
    df["roll3"]      = g.transform(lambda s: s.shift(1).rolling(3,  min_periods=1).mean())
    df["roll5"]      = g.transform(lambda s: s.shift(1).rolling(5,  min_periods=1).mean())
    df["roll10"]     = g.transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    df["exp_avg"]    = g.transform(lambda s: s.shift(1).expanding().mean())
    df["exp_std"]    = g.transform(lambda s: s.shift(1).expanding().std())
    df["roll5_std"]  = g.transform(lambda s: s.shift(1).rolling(5, min_periods=2).std())
    df["roll5_first"]= g.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean().shift(4))
    df["trend_5v5"]  = df["roll5"] - df["roll5_first"]
    df["test_num"]   = df.groupby(["student_id","discipline"]).cumcount()
    gs = df.groupby(["student_id","skill_name"])["y"]
    df["skill_lag1"]    = gs.transform(lambda s: s.shift(1))
    df["skill_roll3"]   = gs.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["skill_exp_avg"] = gs.transform(lambda s: s.shift(1).expanding().mean())
    df["skill_n"]       = gs.transform("cumcount")
    # #O: best ever score in discipline up to (not including) this attempt
    df["best_disc_score"] = g.transform(lambda s: s.shift(1).expanding().max())
    return df.sort_values("ts").reset_index(drop=True)


def add_derived_features(df):
    df["theta_range"]     = df["theta_g"] - df["theta_disc"]
    df["score_momentum"]  = df["surprise5"] * df["n_disc"].clip(upper=20)
    df["gap_bucket"]      = pd.cut(df["days_gap"],
                                   bins=[-1,0,1,3,7,14,30,365],
                                   labels=False).fillna(6).astype(int)
    df["gap_bucket_disc"] = pd.cut(df["days_gap_disc"],
                                   bins=[-1,0,1,3,7,14,30,365],
                                   labels=False).fillna(6).astype(int)
    df["is_cold"]         = (df["n_disc"] < 3).astype(int)
    df["is_cold_skill"]   = (df["n_skill"] < 2).astype(int)
    df["e_hat_residual"]  = df["e_hat"] * 100 - df["exp_avg"].fillna(df["e_hat"] * 100)
    df["ability_diff_gap"]= df["theta_skill"] - df["diff_skill"]
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)
    # #O: non-linear ability and gap consistency
    df["theta_disc_sq"] = df["theta_disc"] ** 2
    df["gap_trend"]     = df["days_gap_disc"] - df["days_gap_skill"]
    return df


# ==============================================================================
# STEP 3b — V4 FEATURES: SESSION + EXAM PROXIMITY (#G, #H)
# ==============================================================================
def add_session_features(feats):
    """
    #G: session_pos (0-indexed position within a study session) and
        session_score_ma (running mean score in session before this test).
    A session is defined as consecutive tests by the same student within 1 hour.
    No lookahead: session_score_ma uses only tests BEFORE the current one in the session.
    """
    df = feats.sort_values(["student_id", "ts"]).copy()
    ts_s = pd.to_datetime(df["ts"]).astype("int64").values / 1e9  # nanoseconds -> seconds

    # Vectorised session boundary detection
    sid_arr = df["student_id"].values
    new_student = np.concatenate([[True], sid_arr[1:] != sid_arr[:-1]])
    gap_arr     = np.concatenate([[0.0], ts_s[1:] - ts_s[:-1]])
    new_session = new_student | (gap_arr > SESSION_GAP_S)

    # Cumulative session counter per student (reset per student via group cumsum)
    df["_new_sess"] = new_session.astype(int)
    df["_sess_id"]  = df.groupby("student_id")["_new_sess"].cumsum()

    # Position in session (0 = first test of session)
    df["session_pos"] = df.groupby(["student_id","_sess_id"]).cumcount()

    # Running mean score in session (shift(1) to exclude current test)
    df["session_score_ma"] = df.groupby(["student_id","_sess_id"])["y"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    # Restore chronological order
    df = df.sort_index()
    result = feats.copy()
    result["session_pos"]      = df["session_pos"].values
    result["session_score_ma"] = df["session_score_ma"].values
    return result


def add_exam_features(feats):
    """
    #H: days_to_june (days until June 1 of current year; 0 if already past June 1)
        in_exam_season (1 if Feb-May = peak UNT preparation months, else 0)
    UNT (Unified National Testing) in Kazakhstan is held in June.
    Students intensify preparation in Feb-May, explaining performance changes.
    """
    dt = pd.to_datetime(feats["ts"])
    year  = dt.dt.year
    month = dt.dt.month

    june1 = pd.to_datetime(
        year.astype(str) + "-06-01", format="%Y-%m-%d"
    )
    # Days until next June 1; clip so past-June = 0 (student already sat the exam)
    days_to_june = (june1 - dt).dt.days.clip(0, 365).astype(float)

    feats = feats.copy()
    feats["days_to_june"]   = days_to_june.values
    feats["in_exam_season"] = ((month >= 2) & (month <= 5)).astype(float).values
    return feats


# ==============================================================================
# STEP 3c — FORGETTING CURVE FEATURES (#K, Strategy C)
# ==============================================================================
def add_forgetting_features(feats):
    """
    #K: Ebbinghaus forgetting curve applied to discipline and skill gaps.

    forget_disc   = exp(-days_gap_disc / TAU_DISC)
        → 1.0 = no gap, 0.0 = very long gap. Captures how much of the student's
          current discipline ability remains accessible after the inactivity period.

    forget_skill  = exp(-days_gap_skill / TAU_SKILL)
        → Faster decay for skill-level memory (more specific, quicker to fade).

    expect_retain = theta_disc * forget_disc
        → Rating-weighted retention: a high-ability student who just returned still
          retains more than a low-ability student with the same gap. This directly
          encodes the cognitive decay mechanism driving the inactivity-gap effect
          observed in Figure (d) of the behavioral analysis.
    """
    df = feats.copy()
    df["forget_disc"]  = np.exp(-df["days_gap_disc"].clip(lower=0) / TAU_DISC)
    df["forget_skill"] = np.exp(-df["days_gap_skill"].clip(lower=0) / TAU_SKILL)
    df["expect_retain"] = df["theta_disc"] * df["forget_disc"]
    return df


# ==============================================================================
# STEP 3d — SCORE VELOCITY FEATURES (#N, V6 new)
# ==============================================================================
def add_velocity_features(df):
    """
    #N: Linear improvement rate over last 5 per-discipline and global attempts.

    score_slope_disc    = slope of last 5 per-discipline pct scores (pp per attempt)
    score_slope_global  = slope of last 5 global pct scores
    attempt_density_30d = attempts in last 30 days (study intensity proxy)
    rd_skill_gap        = rd_skill - rd_disc (per-skill uncertainty above disc baseline)

    All computed with no lookahead: current row not yet included when computing slope.
    """
    df = df.sort_values(["student_id", "ts"]).reset_index(drop=True)

    slope_disc  = np.zeros(len(df), dtype=np.float32)
    slope_glob  = np.zeros(len(df), dtype=np.float32)
    density_30d = np.zeros(len(df), dtype=np.float32)

    disc_hist = {}   # (student_id, discipline) -> list of last-5 pct scores
    glob_hist = {}   # student_id -> list of last-10 (timestamp_s, pct_score)

    for i, row in df.iterrows():
        sid  = row["student_id"]
        disc = row["discipline"]
        ts_s = pd.Timestamp(row["ts"]).timestamp()
        pct  = float(row["y"])

        # slope_disc: slope of last 5 per-discipline scores (before this one)
        key_d  = (sid, disc)
        hist_d = disc_hist.get(key_d, [])
        if len(hist_d) >= 2:
            xs = np.arange(len(hist_d), dtype=float)
            slope_disc[i] = float(np.polyfit(xs, hist_d, 1)[0])
        hist_d = (hist_d + [pct])[-5:]
        disc_hist[key_d] = hist_d

        # slope_glob: slope of last 5 global scores (before this one)
        hist_g = glob_hist.get(sid, [])
        if len(hist_g) >= 2:
            xs = np.arange(len(hist_g), dtype=float)
            slope_glob[i] = float(np.polyfit(xs, [x[1] for x in hist_g], 1)[0])
        # density: attempts in last 30 days (before this one)
        cutoff = ts_s - 30 * 86400
        density_30d[i] = float(sum(1 for t2, _ in hist_g if t2 >= cutoff))
        hist_g = (hist_g + [(ts_s, pct)])[-10:]
        glob_hist[sid] = hist_g

    df["score_slope_disc"]   = slope_disc
    df["score_slope_global"] = slope_glob
    df["attempt_density_30d"] = density_30d
    # rd_skill_gap: difference already available from rating columns
    df["rd_skill_gap"] = df["rd_skill"].values - df["rd_disc"].values
    return df.sort_values("ts").reset_index(drop=True)


# ==============================================================================
# STEP 4 — STATIC CONTENT FEATURES (leakage-safe, per-fold aware)
# ==============================================================================
def add_static_content_features(feats_all, train_mask):
    """
    Discipline/skill difficulty stats computed from train partition only,
    then broadcast to all rows. Drops pre-existing static cols to avoid
    pandas merge collision when called per-fold in rolling-origin.
    """
    static_cols = ["disc_mean_score","disc_std_score",
                   "skill_mean_score","skill_std_score","theta_vs_disc"]
    feats_all = feats_all.drop(
        columns=[c for c in static_cols if c in feats_all.columns]
    )

    train  = feats_all[train_mask]
    g_mean = train["y"].mean()
    g_std  = train["y"].std()

    disc_agg  = train.groupby("discipline")["y"].agg(
        disc_mean_score="mean", disc_std_score="std"
    ).reset_index()
    skill_agg = train.groupby("skill_name")["y"].agg(
        skill_mean_score="mean", skill_std_score="std"
    ).reset_index()

    feats_all = feats_all.merge(disc_agg,  on="discipline",  how="left")
    feats_all = feats_all.merge(skill_agg, on="skill_name",  how="left")
    feats_all["disc_mean_score"]  = feats_all["disc_mean_score"].fillna(g_mean)
    feats_all["disc_std_score"]   = feats_all["disc_std_score"].fillna(g_std)
    feats_all["skill_mean_score"] = feats_all["skill_mean_score"].fillna(g_mean)
    feats_all["skill_std_score"]  = feats_all["skill_std_score"].fillna(g_std)
    feats_all["theta_vs_disc"]    = (feats_all["theta_skill"]
                                     - feats_all["disc_mean_score"] / 100.0)
    return feats_all


# ==============================================================================
# SPLIT
# ==============================================================================
def chrono_split(feats, train_frac=0.70, cal_frac=0.10):
    n  = len(feats)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + cal_frac))
    return feats.iloc[:i1].copy(), feats.iloc[i1:i2].copy(), feats.iloc[i2:].copy()


# ==============================================================================
# MODEL HELPERS
# ==============================================================================
def make_temporal_weights(ts_series, halflife_days=WEIGHT_HALFLIFE):
    """
    #J: exponential recency weighting. Most recent training examples
    get weight ~1.0; examples from `halflife_days` days ago get ~0.5.
    """
    dt  = pd.to_datetime(ts_series)
    end = dt.max()
    age = (end - dt).dt.days.values.astype(float)
    w   = np.exp(-np.log(2.0) * age / halflife_days)
    w   = w / w.mean()   # normalize: mean weight = 1 (no scale distortion)
    return w


def tune_lgb_optuna(X_tr, y_tr, X_val, y_val, sw_tr=None, n_trials=50):
    """
    #I: Optuna tunes LightGBM (50 trials, TPE sampler).
    Objective: maximize R² on validation set.
    Tunes: num_leaves, lr, min_child_samples, feature/bagging fractions, L1/L2.
    """
    if not (HAS_LGB and HAS_OPTUNA):
        log("  [LGB/Optuna absent] Using default LGB params.")
        return LGB_BASE

    def objective(trial):
        params = dict(
            n_estimators        = 5000,
            learning_rate       = trial.suggest_float("lr",   0.01, 0.10, log=True),
            num_leaves          = trial.suggest_int("nl",      63,   511),
            min_child_samples   = trial.suggest_int("mcs",     20,   200),
            feature_fraction    = trial.suggest_float("ff",    0.50, 1.00),
            bagging_fraction    = trial.suggest_float("bf",    0.70, 1.00),
            bagging_freq        = 1,
            reg_alpha           = trial.suggest_float("la",    0.00, 5.00),
            reg_lambda          = trial.suggest_float("ll",    0.00, 5.00),
            n_jobs=-1, random_state=SEED, verbose=-1,
        )
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        return -r2_score(y_val, model.predict(X_val))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    log(f"  Optuna LGB best val R²={-study.best_value:.4f}  params={best}")

    # Build final params dict merging best into LGB_BASE
    final = dict(LGB_BASE)
    final.update({
        "learning_rate":     best["lr"],
        "num_leaves":        best["nl"],
        "min_child_samples": best["mcs"],
        "feature_fraction":  best["ff"],
        "bagging_fraction":  best["bf"],
        "bagging_freq":      1,
        "reg_alpha":         best["la"],
        "reg_lambda":        best["ll"],
    })
    return final


def tune_xgb_optuna(X_tr, y_tr, X_val, y_val, sw_tr=None, n_trials=30):
    """
    Optuna tunes XGBoost (30 trials, TPE sampler).
    Objective: maximize R² on validation set.
    Returns best_params dict with XGBRegressor-compatible keys.
    """
    if not HAS_OPTUNA:
        log("  [Optuna absent] Using default XGB params.")
        return {}

    import xgboost as xgb_lib

    def objective(trial):
        params = dict(
            max_depth        = trial.suggest_int("max_depth", 6, 12),
            learning_rate    = trial.suggest_float("lr", 0.02, 0.08, log=True),
            subsample        = trial.suggest_float("ss", 0.70, 1.00),
            colsample_bytree = trial.suggest_float("cs", 0.60, 1.00),
            min_child_weight = trial.suggest_int("mcw", 3, 20),
            reg_lambda       = trial.suggest_float("rl", 0.5, 5.0),
            reg_alpha        = trial.suggest_float("ra", 0.0, 2.0),
            n_estimators=2000, early_stopping_rounds=50,
            objective="reg:squarederror", tree_method="hist", n_jobs=-1,
            random_state=SEED, verbosity=0,
        )
        m = xgb_lib.XGBRegressor(**params)
        m.fit(X_tr, y_tr, sample_weight=sw_tr,
              eval_set=[(X_val, y_val)], verbose=False)
        return -r2_score(y_val, m.predict(X_val))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    log(f"  Optuna XGB best val R²={-study.best_value:.4f}  params={best}")

    # Remap Optuna short keys to XGBRegressor param names
    return {
        "max_depth":        best["max_depth"],
        "learning_rate":    best["lr"],
        "subsample":        best["ss"],
        "colsample_bytree": best["cs"],
        "min_child_weight": best["mcw"],
        "reg_lambda":       best["rl"],
        "reg_alpha":        best["ra"],
    }


def train_lgb_model(X_tr, y_tr, X_val, y_val, sw_tr=None, params=None):
    if not HAS_LGB:
        return None, None
    p = params if params else LGB_BASE
    model = lgb.LGBMRegressor(**p)
    model.fit(
        X_tr, y_tr,
        sample_weight=sw_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    return model, model.predict(X_val)


def optimal_blend_alpha(xgb_cal, lgb_cal, y_cal):
    """Grid search for XGB blend weight alpha in [0.15, 0.85]."""
    best_r2, best_a = -np.inf, 0.5
    for a in np.arange(0.15, 0.90, 0.05):
        r2 = r2_score(y_cal, a * xgb_cal + (1 - a) * lgb_cal)
        if r2 > best_r2:
            best_r2, best_a = r2, a
    return float(round(best_a, 2))


def blend(xgb_pred, lgb_pred, alpha):
    if lgb_pred is None:
        return xgb_pred
    return alpha * xgb_pred + (1 - alpha) * lgb_pred


# ==============================================================================
# CONFORMAL INTERVALS v5 — Asymmetric CQR (#L) + Fine-grained Mondrian (#M)
# ==============================================================================

def _conformal_quantile(n):
    """Finite-sample corrected quantile level: ceil((n+1)*(1-ALPHA)) / n."""
    return min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0)


def fit_conformal_v5(X_tr, y_tr, X_cal, y_cal, X_te, y_te,
                     lgb_params=None, cal_disc=None, cal_hist=None,
                     te_disc=None, te_hist=None):
    """
    Strategy B (#L): Asymmetric split conformal quantile regression.
      - Train LGB lower (alpha/2) and upper (1-alpha/2) quantile heads.
      - Calibrate each tail independently: separate Q_lo and Q_hi.
      - Asymmetric correction: if model over-predicts weak students (lower bound
        too high), Q_lo grows large and pushes the lower bound down only.

    Strategy A (#M): Fine-grained Mondrian conformal.
      - 4 history bands x 7 discipline bins (top-6 + other) = up to 28 strata.
      - Per-stratum Q_lo / Q_hi computed on calibration set.
      - If stratum n_cal < MIN_STRAT_N, fall back to history-band-only quantile.
      - Dramatically narrows intervals for students with long history
        (band 3, >20 prior tests) where model uncertainty is lower.
    """
    n_cal = len(y_cal)
    y_cal_v = y_cal.values
    y_te_v  = y_te.values

    # ── Quantile model heads (Strategy B) ─────────────────────────────────────
    # V7: wider heads (3rd/97th pct) so raw coverage ~94%, conformal trims to 90%
    qlo_a, qhi_a = QUANTILE_ALPHA / 2, 1 - QUANTILE_ALPHA / 2

    if HAS_LGB and lgb_params:
        p = dict(lgb_params)
        p["n_estimators"] = 2000
        p["verbose"] = -1

        p_lo = dict(p); p_lo.update({"objective": "quantile", "alpha": qlo_a})
        p_hi = dict(p); p_hi.update({"objective": "quantile", "alpha": qhi_a})

        import lightgbm as lgb_lib
        m_lo = lgb_lib.LGBMRegressor(**p_lo)
        m_hi = lgb_lib.LGBMRegressor(**p_hi)
        m_lo.fit(X_tr, y_tr,
                 eval_set=[(X_cal, y_cal)],
                 callbacks=[lgb_lib.early_stopping(30, verbose=False),
                            lgb_lib.log_evaluation(period=-1)])
        m_hi.fit(X_tr, y_tr,
                 eval_set=[(X_cal, y_cal)],
                 callbacks=[lgb_lib.early_stopping(30, verbose=False),
                            lgb_lib.log_evaluation(period=-1)])
        log("  Quantile heads: LightGBM (using Optuna-tuned params)")
    else:
        # Fallback to XGB quantile if LGB unavailable
        XGB_Q = dict(n_estimators=600, learning_rate=0.05, max_depth=8,
                     subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                     reg_lambda=1.5, tree_method="hist", n_jobs=-1,
                     random_state=SEED, verbosity=0)
        m_lo = XGBRegressor(objective="reg:quantileerror", quantile_alpha=qlo_a, **XGB_Q)
        m_hi = XGBRegressor(objective="reg:quantileerror", quantile_alpha=qhi_a, **XGB_Q)
        m_lo.fit(X_tr, y_tr)
        m_hi.fit(X_tr, y_tr)
        log("  Quantile heads: XGBoost (LGB unavailable)")

    lo_cal = m_lo.predict(X_cal)
    hi_cal = m_hi.predict(X_cal)
    lo_te  = m_lo.predict(X_te)
    hi_te  = m_hi.predict(X_te)

    # Asymmetric tail escape scores (Strategy B) — non-negative by construction.
    # Lower tail: how far did lo_cal overshoot ABOVE y_cal? (0 if lo_cal <= y_cal)
    # Upper tail: how far did hi_cal miss BELOW y_cal?     (0 if hi_cal >= y_cal)
    # Using np.maximum ensures Q_lo/Q_hi >= 0 so corrections always widen intervals.
    s_lo_cal = np.maximum(lo_cal - y_cal_v, 0.0)   # >= 0: genuine lower-bound overshoot
    s_hi_cal = np.maximum(y_cal_v - hi_cal, 0.0)   # >= 0: genuine upper-bound miss

    # ── Global asymmetric CQR (no stratification) ─────────────────────────────
    q_level = _conformal_quantile(n_cal)
    Q_lo_g = np.quantile(s_lo_cal, q_level)
    Q_hi_g = np.quantile(s_hi_cal, q_level)

    lo_asym_g = np.clip(lo_te - Q_lo_g, 0, 100)
    hi_asym_g = np.clip(hi_te + Q_hi_g, 0, 100)
    cov_asym_g = float(np.mean((y_te_v >= lo_asym_g) & (y_te_v <= hi_asym_g)))
    wid_asym_g = float(np.mean(hi_asym_g - lo_asym_g))
    log(f"  Asymmetric CQR (global):  coverage={cov_asym_g:.3f}  "
        f"width={wid_asym_g:.1f}  Q_lo={Q_lo_g:.2f}  Q_hi={Q_hi_g:.2f}")

    # ── Fine-grained Mondrian (Strategy A) ────────────────────────────────────
    # History band: 0=[1-4], 1=[5-10], 2=[11-20], 3=[>20]
    def hist_band(n_arr):
        b = np.zeros(len(n_arr), dtype=int)
        b[n_arr >= 5]  = 1
        b[n_arr >= 11] = 2
        b[n_arr >= 21] = 3
        return b

    # Discipline bin: top-6 discipline codes become 0-5, rest become 6
    def disc_bin(disc_arr, top_codes):
        b = np.full(len(disc_arr), 6, dtype=int)  # default = "other"
        for rank, code in enumerate(top_codes):
            b[disc_arr == code] = rank
        return b

    lo_m_te = lo_te.copy()
    hi_m_te = hi_te.copy()

    strat_stats = []

    if cal_disc is not None and cal_hist is not None:
        hb_cal = hist_band(cal_hist)
        hb_te  = hist_band(te_hist)

        # Identify top-6 discipline codes by calibration frequency
        from collections import Counter
        top_codes = [code for code, _ in Counter(cal_disc).most_common(6)]

        db_cal = disc_bin(cal_disc, top_codes)
        db_te  = disc_bin(te_disc,  top_codes)

        # Pre-compute fallback quantiles per history band
        fallback_lo = {}
        fallback_hi = {}
        for hb in range(4):
            m_hb = hb_cal == hb
            if m_hb.sum() >= MIN_STRAT_N:
                fallback_lo[hb] = np.quantile(s_lo_cal[m_hb], _conformal_quantile(m_hb.sum()))
                fallback_hi[hb] = np.quantile(s_hi_cal[m_hb], _conformal_quantile(m_hb.sum()))
            else:
                fallback_lo[hb] = Q_lo_g
                fallback_hi[hb] = Q_hi_g

        for hb in range(4):
            for db in range(7):
                mc = (hb_cal == hb) & (db_cal == db)
                mt = (hb_te  == hb) & (db_te  == db)
                if mt.sum() == 0:
                    continue
                if mc.sum() >= MIN_STRAT_N:
                    q_lo_s = np.quantile(s_lo_cal[mc], _conformal_quantile(mc.sum()))
                    q_hi_s = np.quantile(s_hi_cal[mc], _conformal_quantile(mc.sum()))
                else:
                    # Fall back to history-band quantile
                    q_lo_s = fallback_lo[hb]
                    q_hi_s = fallback_hi[hb]

                lo_m_te[mt] = np.clip(lo_te[mt] - q_lo_s, 0, 100)
                hi_m_te[mt] = np.clip(hi_te[mt] + q_hi_s, 0, 100)
                cov_s = float(np.mean(
                    (y_te_v[mt] >= lo_m_te[mt]) & (y_te_v[mt] <= hi_m_te[mt])
                ))
                wid_s = float(np.mean(hi_m_te[mt] - lo_m_te[mt]))
                strat_stats.append({
                    "hist_band": hb, "disc_bin": db,
                    "n_cal": int(mc.sum()), "n_te": int(mt.sum()),
                    "coverage": cov_s, "width": wid_s,
                })
    else:
        # No stratum info: fall back to global asymmetric
        lo_m_te = lo_asym_g
        hi_m_te = hi_asym_g

    cov_m = float(np.mean((y_te_v >= lo_m_te) & (y_te_v <= hi_m_te)))
    wid_m = float(np.mean(hi_m_te - lo_m_te))

    # Width breakdown by history band
    if cal_hist is not None:
        log(f"  Fine-grained Mondrian:    coverage={cov_m:.3f}  width={wid_m:.1f}")
        for hb, label in [(0,"1-4"),(1,"5-10"),(2,"11-20"),(3,">20")]:
            mt_hb = hist_band(te_hist) == hb
            if mt_hb.sum() > 0:
                wid_hb = float(np.mean(hi_m_te[mt_hb] - lo_m_te[mt_hb]))
                cov_hb = float(np.mean(
                    (y_te_v[mt_hb] >= lo_m_te[mt_hb]) & (y_te_v[mt_hb] <= hi_m_te[mt_hb])
                ))
                log(f"    hist_band={label}: n={mt_hb.sum():>7,}  "
                    f"cov={cov_hb:.3f}  width={wid_hb:.1f}")

    return {
        "asym_cqr_coverage": cov_asym_g, "asym_cqr_width": wid_asym_g,
        "asym_cqr_q_lo": float(Q_lo_g),  "asym_cqr_q_hi": float(Q_hi_g),
        "mondrian_fg_coverage": cov_m,   "mondrian_fg_width": wid_m,
        "strat_stats": strat_stats,
    }


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    t_start = time.time()
    log(f"  LightGBM available : {HAS_LGB}")
    log(f"  Optuna available   : {HAS_OPTUNA}")

    # STEP 1
    df = load_and_clean()

    # STEP 2: Rating features v4 (with global rolling stats)
    hdr("STEP 2: RATING FEATURES V4")
    feats = build_rating_features_v4(df)

    # STEP 3: Aggregate + derived + v4 features
    hdr("STEP 3: AGGREGATE + DERIVED + V4 FEATURES")
    feats = build_aggregate_features(feats)
    feats = add_derived_features(feats)
    feats = add_session_features(feats)      # #G: session context
    feats = add_exam_features(feats)         # #H: UNT exam proximity
    feats = add_forgetting_features(feats)   # #K: forgetting curve (Strategy C)
    feats = add_velocity_features(feats)     # #N: score velocity (V6)
    log(f"  Feature matrix: {len(feats):,} rows")

    # STEP 4: Chronological split + static content features
    hdr("STEP 4: SPLIT + STATIC CONTENT FEATURES")
    n_total   = len(feats)
    n_tr      = int(n_total * 0.70)
    n_cal     = int(n_total * 0.10)
    train_mask = np.zeros(n_total, dtype=bool)
    train_mask[:n_tr] = True

    feats = add_static_content_features(feats, train_mask)

    tr  = feats.iloc[:n_tr].copy()
    cal = feats.iloc[n_tr:n_tr+n_cal].copy()
    te  = feats.iloc[n_tr+n_cal:].copy()

    log(f"  Train: {len(tr):,}  Cal: {len(cal):,}  Test: {len(te):,}")
    log(f"  Train dates: {pd.to_datetime(tr['ts']).min().date()} -> "
        f"{pd.to_datetime(tr['ts']).max().date()}")
    log(f"  Test  dates: {pd.to_datetime(te['ts']).min().date()} -> "
        f"{pd.to_datetime(te['ts']).max().date()}")

    v7_feats = [f for f in GRACE_V7_FEATURES if f in feats.columns]
    v6_feats = [f for f in GRACE_V6_FEATURES if f in feats.columns]
    v5_feats = [f for f in GRACE_V5_FEATURES if f in feats.columns]
    v4_feats = [f for f in GRACE_V4_FEATURES if f in feats.columns]
    v2_feats = [f for f in GRACE_V2_FEATURES if f in feats.columns]
    rec("FEAT_N_V7", len(v7_feats))
    rec("FEAT_N_V6", len(v6_feats))
    rec("FEAT_N_V5", len(v5_feats))
    rec("FEAT_N_V4", len(v4_feats))
    rec("FEAT_N_V2", len(v2_feats))
    log(f"  V7 features: {len(v7_feats)}  (V6 base: {len(v6_feats)}, V5: {len(v5_feats)}, V2 ref: {len(v2_feats)})")
    log(f"  New V7 features (#O): {[f for f in GRACE_V7_EXTRA if f in feats.columns]}")
    log(f"  Dropped: {[f for f in GRACE_V7_DROP if f in feats.columns]}")

    X_tr  = tr[v7_feats].fillna(-999)
    X_cal = cal[v7_feats].fillna(-999)
    X_te  = te[v7_feats].fillna(-999)
    y_tr, y_cal, y_te = tr["y"], cal["y"], te["y"]

    # #J: temporal sample weights for training
    sw_tr = make_temporal_weights(tr["ts"], halflife_days=WEIGHT_HALFLIFE)
    log(f"  Sample weight range: [{sw_tr.min():.3f}, {sw_tr.max():.3f}]  "
        f"mean={sw_tr.mean():.3f}")

    # ── EXPERIMENT 1: Leakage audit (reference, unchanged) ────────────────────
    hdr("EXPERIMENT 1: LEAKAGE AUDIT (unchanged reference)")
    rec("E1_INFLATION_R2",    "+0.349")
    rec("E1_CLEAN_CHRONO_R2", "0.444")
    log("  (Factorial audit unchanged — see run_log_v6 for full breakdown)")

    # ── EXPERIMENT 2: GRACE V4 vs baselines ──────────────────────────────────
    hdr("EXPERIMENT 2: GRACE V4 vs BASELINES (chrono 70/10/20)")

    # 2a. XGBoost — Optuna-tuned (30 trials)
    log("\n  2a. XGBoost (Optuna-tuned, 30 trials)")
    xgb_best = tune_xgb_optuna(X_tr, y_tr, X_cal, y_cal, sw_tr=sw_tr, n_trials=30)
    xgb_params = dict(XGB_BASE)
    xgb_params.update(xgb_best)
    xgb = XGBRegressor(**xgb_params)
    xgb.fit(X_tr, y_tr, sample_weight=sw_tr,
            eval_set=[(X_cal, y_cal)], verbose=False)
    xgb_pred_cal = xgb.predict(X_cal)
    xgb_pred_te  = xgb.predict(X_te)
    r2_xgb  = r2_score(y_te,  xgb_pred_te)
    mae_xgb = mean_absolute_error(y_te, xgb_pred_te)
    rec("E2_XGB_R2",  f"{r2_xgb:.3f}")
    rec("E2_XGB_MAE", f"{mae_xgb:.2f}")
    log(f"  XGBoost v6: R²={r2_xgb:.4f}  MAE={mae_xgb:.2f}")

    # 2b. LightGBM — Optuna-tuned (#I)
    log("\n  2b. LightGBM (Optuna-tuned)")
    lgb_params = tune_lgb_optuna(X_tr, y_tr, X_cal, y_cal,
                                  sw_tr=sw_tr, n_trials=20)
    lgb_model, lgb_pred_cal = train_lgb_model(
        X_tr, y_tr, X_cal, y_cal, sw_tr=sw_tr, params=lgb_params
    )
    lgb_pred_te = lgb_model.predict(X_te) if lgb_model else None

    if lgb_pred_te is not None:
        r2_lgb  = r2_score(y_te,  lgb_pred_te)
        mae_lgb = mean_absolute_error(y_te, lgb_pred_te)
        rec("E2_LGB_R2",  f"{r2_lgb:.3f}")
        rec("E2_LGB_MAE", f"{mae_lgb:.2f}")
        log(f"  LightGBM v6: R²={r2_lgb:.4f}  MAE={mae_lgb:.2f}")

        # Optimal blend
        alpha = optimal_blend_alpha(xgb_pred_cal, lgb_pred_cal, y_cal)
        final_pred_te  = blend(xgb_pred_te,  lgb_pred_te,  alpha)
        final_pred_cal = blend(xgb_pred_cal, lgb_pred_cal, alpha)
    else:
        alpha = 1.0
        final_pred_te  = xgb_pred_te
        final_pred_cal = xgb_pred_cal

    r2_v5  = r2_score(y_te,  final_pred_te)
    mae_v5 = mean_absolute_error(y_te, final_pred_te)
    rec("E2_BLEND_ALPHA",    f"{alpha:.2f}")
    rec("E2_GRACE_V7_R2",    f"{r2_v5:.3f}")
    rec("E2_GRACE_V7_MAE",   f"{mae_v5:.2f}")
    rec("E2_GRACE_V7_NTEST", f"{len(y_te):,}")
    log(f"\n  GRACE V7 ensemble (alpha_xgb={alpha:.2f}): "
        f"R²={r2_v5:.4f}  MAE={mae_v5:.2f}")
    rec("E2_DR2_VS_V6",  f"{r2_v5 - 0.556:+.3f}")
    rec("E2_DMAE_VS_V6", f"{mae_v5 - 11.70:+.2f}")
    log(f"  vs v6 (0.556, 11.70): dR²={r2_v5-0.556:+.4f}  dMAE={mae_v5-11.70:+.2f}")
    rec("E2_DR2_VS_V4",  f"{r2_v5 - 0.557:+.3f}")
    log(f"  vs v4 (0.557, 11.70): dR²={r2_v5-0.557:+.4f}  dMAE={mae_v5-11.70:+.2f}")
    rec("E2_DR2_VS_V2",  f"{r2_v5 - 0.544:+.3f}")
    log(f"  vs v2 (0.544, 11.86): dR²={r2_v5-0.544:+.4f}  dMAE={mae_v5-11.86:+.2f}")

    # 2c. V2 features only (ablation reference, Optuna-tuned params)
    log(f"\n  2c. V2 features only ({len(v2_feats)} feat, XGB Optuna) -- ablation ref")
    xgb_v2 = XGBRegressor(**xgb_params)
    xgb_v2.fit(tr[v2_feats].fillna(-999), y_tr,
               sample_weight=sw_tr,
               eval_set=[(cal[v2_feats].fillna(-999), y_cal)], verbose=False)
    r2_v2_here  = r2_score(y_te,  xgb_v2.predict(te[v2_feats].fillna(-999)))
    mae_v2_here = mean_absolute_error(y_te, xgb_v2.predict(te[v2_feats].fillna(-999)))
    rec("E2_V2_HERE_R2",  f"{r2_v2_here:.3f}")
    rec("E2_V2_HERE_MAE", f"{mae_v2_here:.2f}")
    log(f"  V2 features (this run): R²={r2_v2_here:.4f}  MAE={mae_v2_here:.2f}")

    r2_tr_v5  = r2_score(y_tr, xgb.predict(X_tr))
    mae_tr_v5 = mean_absolute_error(y_tr, xgb.predict(X_tr))
    rec("E2_GRACE_V7_TRAIN_R2",  f"{r2_tr_v5:.3f}")
    rec("E2_GRACE_V7_TRAIN_MAE", f"{mae_tr_v5:.2f}")
    rec("E2_GRACE_V7_GAP",       f"{r2_tr_v5 - r2_v5:+.3f}")
    log(f"  Train: R²={r2_tr_v5:.3f}  gap={r2_tr_v5-r2_v5:+.3f}")

    # ── EXPERIMENT 3: Rolling-origin V5 ──────────────────────────────────────
    hdr("EXPERIMENT 3: ROLLING-ORIGIN V7 (5 folds, 2021-2025)")
    RO_FOLDS = [(2021,), (2022,), (2023,), (2024,), (2025,)]
    ro_r2s, ro_maes = [], []
    ro_base_r2s, ro_base_maes = [], []

    for fi, (test_yr,) in enumerate(RO_FOLDS, 1):
        ts_col = pd.to_datetime(feats["ts"]).dt.year
        f_tr_raw = feats[ts_col < test_yr].copy()
        f_te_raw = feats[ts_col == test_yr].copy()
        if len(f_tr_raw) < 1000 or len(f_te_raw) < 100:
            continue

        n_ftr = len(f_tr_raw)
        f_all = pd.concat([f_tr_raw, f_te_raw], ignore_index=True)
        f_all = add_static_content_features(
            f_all, np.arange(len(f_all)) < n_ftr
        )
        f_tr_fold = f_all.iloc[:n_ftr]
        f_te_fold = f_all.iloc[n_ftr:]

        av    = [ff for ff in v7_feats if ff in f_tr_fold.columns]
        Xf_tr = f_tr_fold[av].fillna(-999)
        Xf_te = f_te_fold[av].fillna(-999)
        yf_tr = f_tr_fold["y"]
        yf_te = f_te_fold["y"]

        sw_f = make_temporal_weights(f_tr_fold["ts"])
        cal_cut = int(0.90 * len(Xf_tr))
        Xft, yft = Xf_tr.iloc[:cal_cut], yf_tr.iloc[:cal_cut]
        Xfc, yfc = Xf_tr.iloc[cal_cut:], yf_tr.iloc[cal_cut:]
        sw_ft = sw_f[:cal_cut]

        # Use fixed XGB params for rolling-origin speed
        mf = XGBRegressor(**XGB_BASE)
        mf.fit(Xft, yft, sample_weight=sw_ft,
               eval_set=[(Xfc, yfc)], verbose=False)
        r2_f  = r2_score(yf_te, mf.predict(Xf_te))
        mae_f = mean_absolute_error(yf_te, mf.predict(Xf_te))
        ro_r2s.append(r2_f)
        ro_maes.append(mae_f)
        rec(f"E3_F{fi}_GRACE_V7_R2",  f"{r2_f:.3f}")
        rec(f"E3_F{fi}_GRACE_V7_MAE", f"{mae_f:.2f}")
        rec(f"E3_F{fi}_NTRAIN",       f"{len(Xft):,}")
        rec(f"E3_F{fi}_NTEST",        f"{len(Xf_te):,}")
        log(f"  Fold {fi} (train<{test_yr}->{test_yr}): "
            f"R²={r2_f:.3f}  MAE={mae_f:.2f}  "
            f"n_train={len(Xft):,}  n_test={len(Xf_te):,}")

        # V2-feature baseline for fair comparison (same fold, XGB_BASE, no Optuna)
        av2_fold = [ff for ff in v2_feats if ff in f_tr_fold.columns]
        if av2_fold:
            Xft_v2 = f_tr_fold.iloc[:cal_cut][av2_fold].fillna(-999)
            Xfc_v2 = f_tr_fold.iloc[cal_cut:][av2_fold].fillna(-999)
            Xte_v2 = f_te_fold[av2_fold].fillna(-999)
            mf_base = XGBRegressor(**XGB_BASE)
            mf_base.fit(Xft_v2, yft, sample_weight=sw_ft,
                        eval_set=[(Xfc_v2, yfc)], verbose=False)
            r2_base_f  = r2_score(yf_te, mf_base.predict(Xte_v2))
            mae_base_f = mean_absolute_error(yf_te, mf_base.predict(Xte_v2))
            ro_base_r2s.append(r2_base_f)
            ro_base_maes.append(mae_base_f)
            rec(f"E3_F{fi}_BASE_R2",  f"{r2_base_f:.3f}")
            rec(f"E3_F{fi}_BASE_MAE", f"{mae_base_f:.2f}")
            log(f"  Fold {fi} V2 baseline (46 feat, XGB_BASE):  "
                f"R²={r2_base_f:.3f}  MAE={mae_base_f:.2f}")

    _results["E3_GRACE_R2_LIST"] = [round(r, 3) for r in ro_r2s]
    if ro_base_r2s:
        _results["E3_BASE_R2_LIST"] = [round(r, 3) for r in ro_base_r2s]
        ro_base_mean = float(np.mean(ro_base_r2s))
        ro_base_std  = float(np.std(ro_base_r2s))
        rec("E3_BASE_MEAN_R2", f"{ro_base_mean:.3f}")
        rec("E3_BASE_SD_R2",   f"{ro_base_std:.3f}")
        log(f"  V2 Baseline RO: {ro_base_mean:.3f} +/- {ro_base_std:.3f}  "
            f"({len(ro_base_r2s)} folds)")

    ro_mean = float(np.mean(ro_r2s))
    ro_std  = float(np.std(ro_r2s))
    ro_mae_mean = float(np.mean(ro_maes))
    rec("E3_GRACE_V7_MEAN_R2",  f"{ro_mean:.3f}")
    rec("E3_GRACE_V7_SD_R2",    f"{ro_std:.3f}")
    rec("E3_GRACE_V7_MEAN_MAE", f"{ro_mae_mean:.2f}")
    log(f"\n  Rolling-origin V7: {ro_mean:.3f} +/- {ro_std:.3f}  "
        f"MAE {ro_mae_mean:.2f}  ({len(ro_r2s)} folds)")
    log(f"  vs v6: 0.514 +/- 0.052  dR²={ro_mean-0.514:+.3f}")
    log(f"  vs v4: 0.514 +/- 0.050  dR²={ro_mean-0.514:+.3f}")

    # ── EXPERIMENT 4: Ablation — V4+V5 additions ─────────────────────────────
    hdr("EXPERIMENT 4: ABLATION -- V4+V5+V6+V7 ADDITIONS")
    ablations = {
        "V7 full (78 feat, XGB)":     v7_feats,
        "- new features (#O)":        [f for f in v7_feats
                                        if f not in set(GRACE_V7_EXTRA)],
        "- score velocity (#N)":      [f for f in v7_feats
                                        if f not in set(GRACE_V6_EXTRA) - GRACE_V7_DROP],
        "- forgetting curve (#K)":    [f for f in v7_feats
                                        if f not in set(GRACE_V5_EXTRA)],
        "- global rolling (#F)":      [f for f in v7_feats
                                        if f not in {"global_lag1","global_roll3",
                                                     "global_roll5","global_roll10",
                                                     "global_std5","ability_gap_global"}],
        "- session features (#G)":    [f for f in v7_feats
                                        if f not in {"session_pos","session_score_ma"}],
        "- all V4+V5+V6+V7 extras":   [f for f in v7_feats
                                        if f not in set(GRACE_V4_EXTRA + GRACE_V5_EXTRA
                                                        + GRACE_V6_EXTRA + GRACE_V7_EXTRA)],
        "V2 features only (46)":      v2_feats,
    }
    e4_ablation = []
    for name, feat_sub in ablations.items():
        if not feat_sub:
            continue
        Xa_tr  = tr[feat_sub].fillna(-999)
        Xa_cal = cal[feat_sub].fillna(-999)
        Xa_te  = te[feat_sub].fillna(-999)
        ma = XGBRegressor(**xgb_params)
        ma.fit(Xa_tr, y_tr, sample_weight=sw_tr,
               eval_set=[(Xa_cal, y_cal)], verbose=False)
        r2_a  = r2_score(y_te,  ma.predict(Xa_te))
        mae_a = mean_absolute_error(y_te, ma.predict(Xa_te))
        dr    = r2_a - r2_xgb
        log(f"  {name:<42s}: R²={r2_a:.4f} ({dr:+.4f} vs V7 XGB)  MAE={mae_a:.2f}")
        e4_ablation.append({"name": name, "r2": round(r2_a, 4),
                            "mae": round(mae_a, 2), "delta_r2": round(dr, 4),
                            "n_feats": len(feat_sub)})
    _results["E4_ABLATION"] = e4_ablation

    # ── EXPERIMENT 5: Score ranges — Asymmetric CQR + Fine-grained Mondrian ──
    hdr("EXPERIMENT 5: SCORE RANGES V7 (Wider CQR 3rd/97th + Fine-grained Mondrian)")
    conf = fit_conformal_v5(   # function retained from v5; fix applied inside
        X_tr, y_tr, X_cal, y_cal, X_te, y_te,
        lgb_params=lgb_params if HAS_LGB else None,
        cal_disc=cal["disc_code"].values if "disc_code" in cal.columns else None,
        cal_hist=cal["n_disc"].values    if "n_disc"    in cal.columns else None,
        te_disc =te["disc_code"].values  if "disc_code" in te.columns  else None,
        te_hist =te["n_disc"].values     if "n_disc"    in te.columns  else None,
    )
    rec("E5_ASYM_CQR_COV",    f"{conf['asym_cqr_coverage']:.3f}")
    rec("E5_ASYM_CQR_WIDTH",  f"{conf['asym_cqr_width']:.1f}")
    rec("E5_ASYM_CQR_Q_LO",   f"{conf['asym_cqr_q_lo']:.3f}")
    rec("E5_ASYM_CQR_Q_HI",   f"{conf['asym_cqr_q_hi']:.3f}")
    rec("E5_MONDRIAN_FG_COV",  f"{conf['mondrian_fg_coverage']:.3f}")
    rec("E5_MONDRIAN_FG_WIDTH",f"{conf['mondrian_fg_width']:.1f}")
    log(f"\n  vs v6 (coverage=0.859, width=44.4 pp, Q_lo=Q_hi=0):")
    log(f"  Asym CQR  : coverage={conf['asym_cqr_coverage']:.3f}  "
        f"width={conf['asym_cqr_width']:.1f}  "
        f"(dCov={conf['asym_cqr_coverage']-0.859:+.3f}  dWidth={conf['asym_cqr_width']-44.4:+.1f})")
    log(f"  Mondrian FG: coverage={conf['mondrian_fg_coverage']:.3f}  "
        f"width={conf['mondrian_fg_width']:.1f}  "
        f"(dCov={conf['mondrian_fg_coverage']-0.860:+.3f}  dWidth={conf['mondrian_fg_width']-44.4:+.1f})")
    log(f"  vs v4 Mondrian (0.889, 46.6 pp): dCov={conf['asym_cqr_coverage']-0.889:+.3f}  "
        f"dWidth={conf['asym_cqr_width']-46.6:+.1f}")

    # ── Export E5_HISTBAND_STATS (aggregate strat_stats by hist_band) ─────────
    _hb_agg = {}
    for _row in conf.get("strat_stats", []):
        _hb = _row["hist_band"]
        if _hb not in _hb_agg:
            _hb_agg[_hb] = {"n": 0, "cov_wsum": 0.0, "wid_wsum": 0.0}
        _n = _row["n_te"]
        _hb_agg[_hb]["n"]        += _n
        _hb_agg[_hb]["cov_wsum"] += _row["coverage"] * _n
        _hb_agg[_hb]["wid_wsum"] += _row["width"]    * _n
    _band_labels = {0: "1-4", 1: "5-10", 2: "11-20", 3: ">20"}
    e5_histband = []
    for _hb in range(4):
        if _hb in _hb_agg:
            _s = _hb_agg[_hb]
            e5_histband.append({
                "band": _hb, "label": _band_labels[_hb],
                "n": _s["n"],
                "coverage": round(_s["cov_wsum"] / _s["n"], 3),
                "width":    round(_s["wid_wsum"] / _s["n"], 1),
            })
    _results["E5_HISTBAND_STATS"] = e5_histband
    log(f"\n  E5_HISTBAND_STATS: {len(e5_histband)} bands")
    for _r in e5_histband:
        log(f"    band={_r['label']:>5s}: n={_r['n']:>7,}  "
            f"cov={_r['coverage']:.3f}  width={_r['width']:.1f}")

    # ── EXPERIMENT 6: Per-student evaluation ─────────────────────────────────
    hdr("EXPERIMENT 6: PER-STUDENT EVALUATION")
    te_eval = te[["student_id","y","n_disc"]].copy()
    te_eval["pred"] = final_pred_te
    ps = []
    for _, grp in te_eval.groupby("student_id"):
        if len(grp) >= 3:
            ps.append(r2_score(grp["y"], grp["pred"]))
    ps = np.array(ps)
    pct_pos   = float(np.mean(ps > 0) * 100)
    median_r2 = float(np.median(ps))
    pct_neg   = float(np.mean(ps < 0) * 100)
    rec("E6_PERSTUDENT_N",         len(ps))
    rec("E6_PERSTUDENT_MEDIAN_R2", f"{median_r2:.3f}")
    rec("E6_PERSTUDENT_PCT_POS",   f"{pct_pos:.1f}")
    rec("E6_PERSTUDENT_PCT_NEG",   f"{pct_neg:.1f}")
    log(f"  Per-student (n>=3): {len(ps):,}  "
        f"median R²={median_r2:.3f}  "
        f"{pct_pos:.1f}% positive  {pct_neg:.1f}% negative")
    log(f"  vs v2: median R²=0.283  77.2% pos  22.8% neg")

    te_eval["decile"] = pd.qcut(te_eval["y"], q=10, labels=False, duplicates="drop")
    bias_low  = float((te_eval[te_eval["decile"]==0]["pred"]
                       - te_eval[te_eval["decile"]==0]["y"]).mean())
    bias_high = float((te_eval[te_eval["decile"]==9]["pred"]
                       - te_eval[te_eval["decile"]==9]["y"]).mean())
    rec("E6_BIAS_LOW_DECILE",  f"{bias_low:+.1f}")
    rec("E6_BIAS_HIGH_DECILE", f"{bias_high:+.1f}")
    log(f"  Bias decile 1 (low): {bias_low:+.1f}  decile 10 (high): {bias_high:+.1f}")

    # ── EXPERIMENT 7: History depth ───────────────────────────────────────────
    hdr("EXPERIMENT 7: HISTORY DEPTH ANALYSIS")
    te_eval2 = te[["y","n_disc"]].copy()
    te_eval2["pred"] = final_pred_te
    for lo, hi, label in [(1,4,"1-4"),(5,10,"5-10"),(11,20,"11-20"),(21,999,">20")]:
        mask = (te_eval2["n_disc"] >= lo) & (te_eval2["n_disc"] <= hi)
        sub  = te_eval2[mask]
        if len(sub) < 10:
            continue
        r2_h  = r2_score(sub["y"], sub["pred"])
        mae_h = mean_absolute_error(sub["y"], sub["pred"])
        log(f"  n={label:<6s}: {len(sub):>7,}  R²={r2_h:.3f}  MAE={mae_h:.2f}")

    # ── EXPERIMENT 8: Per-discipline R² ──────────────────────────────────────
    hdr("EXPERIMENT 8: PER-DISCIPLINE R² (GRACE vs V2 baseline)")
    te_disc_eval = te[["discipline", "y"]].copy()
    te_disc_eval["pred_grace"] = final_pred_te
    te_disc_eval["pred_v2"]    = xgb_v2.predict(te[v2_feats].fillna(-999))
    e8_per_disc = []
    for _disc, _grp in te_disc_eval.groupby("discipline"):
        if len(_grp) < 200:
            continue
        _r2_g  = r2_score(_grp["y"], _grp["pred_grace"])
        _r2_b  = r2_score(_grp["y"], _grp["pred_v2"])
        _delta = _r2_g - _r2_b
        e8_per_disc.append({
            "discipline": _disc,
            "n": len(_grp),
            "r2_grace": round(_r2_g, 3),
            "r2_base":  round(_r2_b, 3),
            "delta":    round(_delta, 3),
        })
        log(f"  {_disc:<40s}: n={len(_grp):>7,}  "
            f"R²_GRACE={_r2_g:.3f}  R²_V2={_r2_b:.3f}  Δ={_delta:+.3f}")
    e8_per_disc.sort(key=lambda x: x["r2_grace"], reverse=True)
    _results["E8_PER_DISC"] = e8_per_disc
    if e8_per_disc:
        _best  = e8_per_disc[0]
        _worst = e8_per_disc[-1]
        rec("E8_MAX_DISC", f"{_best['discipline']}  R²={_best['r2_grace']:.3f}")
        rec("E8_MIN_DISC", f"{_worst['discipline']}  R²={_worst['r2_grace']:.3f}")
        log(f"\n  Best  discipline: {_best['discipline']}  R²={_best['r2_grace']:.3f}")
        log(f"  Worst discipline: {_worst['discipline']}  R²={_worst['r2_grace']:.3f}")

    # ── FEATURE IMPORTANCE (LGB if available, else XGB) ──────────────────────
    hdr("TOP-20 FEATURE IMPORTANCE")
    if lgb_model is not None:
        imp = pd.Series(lgb_model.feature_importances_, index=v7_feats).sort_values(ascending=False)
        log("  Source: LightGBM (split importance)")
    else:
        imp = pd.Series(xgb.feature_importances_, index=v7_feats).sort_values(ascending=False)
        log("  Source: XGBoost (gain importance)")
    total = imp.sum()
    for feat, val in imp.head(20).items():
        bar = "#" * int(40 * val / max(imp.iloc[0], 1))
        log(f"  {feat:<30s} {val/total:>6.1%}  {bar}")

    # Export top-20 feature importance
    e5_feat_imp = [[feat, round(100.0 * val / max(total, 1), 2)]
                   for feat, val in imp.head(20).items()]
    _results["E5_FEAT_IMP"] = e5_feat_imp
    log(f"\n  E5_FEAT_IMP exported: {len(e5_feat_imp)} features (top-20)")

    # V7 (#O) + V6 (#N) + V5 (#K) + V4 features importance rank
    log("\n  V7 new features (#O) ranks:")
    for f in GRACE_V7_EXTRA:
        if f in imp.index:
            rank = list(imp.index).index(f) + 1
            pct  = imp[f] / total
            log(f"    {f:<30s}  rank #{rank:<3d}  {pct:.1%}")
    log("\n  V6 velocity features (#N) ranks:")
    for f in GRACE_V6_EXTRA:
        if f in imp.index:
            rank = list(imp.index).index(f) + 1
            pct  = imp[f] / total
            log(f"    {f:<30s}  rank #{rank:<3d}  {pct:.1%}")
    log("\n  V5 forgetting features (#K) ranks:")
    for f in GRACE_V5_EXTRA:
        if f in imp.index:
            rank = list(imp.index).index(f) + 1
            pct  = imp[f] / total
            log(f"    {f:<30s}  rank #{rank:<3d}  {pct:.1%}")
    log("\n  V4 feature ranks:")
    for f in GRACE_V4_EXTRA:
        if f in imp.index:
            rank = list(imp.index).index(f) + 1
            pct  = imp[f] / total
            log(f"    {f:<30s}  rank #{rank:<3d}  {pct:.1%}")

    # ── SAVE ─────────────────────────────────────────────────────────────────
    elapsed = (time.time() - t_start) / 60
    rec("RUNTIME_MIN", f"{elapsed:.1f}")

    hdr("PIPELINE COMPLETE")
    log(f"  Runtime:            {elapsed:.1f} min")
    log(f"  GRACE V7 R²:        {r2_v5:.4f}  (v6: 0.556  v4: 0.557)")
    log(f"  GRACE V7 MAE:       {mae_v5:.2f}  (v6: 11.70  v4: 11.70)")
    log(f"  Rolling-origin:     {ro_mean:.3f} +/- {ro_std:.3f}  (v6: 0.514+/-0.052  v4: 0.514+/-0.050)")
    log(f"  Asym CQR coverage:  {conf['asym_cqr_coverage']:.3f}  width={conf['asym_cqr_width']:.1f} pp  "
        f"(v6: 0.859/44.4  v4: 0.889/46.6)")
    log(f"  Mondrian FG cov:    {conf['mondrian_fg_coverage']:.3f}  width={conf['mondrian_fg_width']:.1f} pp")
    log(f"  Q_lo (conformal):   {conf['asym_cqr_q_lo']:.3f}  (target > 0)")
    log(f"  Q_hi (conformal):   {conf['asym_cqr_q_hi']:.3f}  (target > 0)")
    log(f"  dR² vs v6:          {r2_v5-0.556:+.4f}")
    log(f"  dR² vs v4:          {r2_v5-0.557:+.4f}")
    log(f"  dR² vs v2:          {r2_v5-0.544:+.4f}")

    RES_PATH.write_text(json.dumps(_results, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG_PATH.write_text("\n".join(_log_lines), encoding="utf-8")
    log(f"  Saved: {RES_PATH}")
    log(f"  Log:   {LOG_PATH}")


if __name__ == "__main__":
    main()
