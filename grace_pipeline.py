"""
GRACE — Glicko-informed Rating-Augmented Conformal Estimation
Reference implementation for continuous-score student performance prediction.

Layers:
  1. Streaming multi-level rating state (ability + difficulty + rating deviation
     with inactivity time-decay). Leakage-free by construction: features for
     attempt t use only state accumulated strictly before t.
  2. Gradient-boosted point + quantile regressors (XGBoost).
  3. RD-stratified (Mondrian) conformalized quantile regression.

Also includes: clean aggregate baseline features, rolling-origin evaluation,
and a leakage-audit mode reproducing the flawed pipeline for E2.

Usage:
    python grace_pipeline.py --data test_results.csv --experiment e1_e3
    python grace_pipeline.py --data test_results.csv --experiment rolling
    python grace_pipeline.py --data test_results.csv --experiment leakage_audit

Expected columns (rename in CONFIG): StudentClientId, Discipline, SkillName,
TestTypeCategoryName, Score, MaxSkillScore, DateTime.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
COL = dict(student="student_id", disc="discipline", skill="skill_name",
           ttype="test_type_category_id", score="score", maxscore="max_score",
           ts="test_datetime")
SEED = 42
ALPHA = 0.10           # 90% prediction intervals

# Rating hyperparameters — tune ONCE on 2018–2020, then freeze (Part D, risks)
K0 = 0.35              # base gain
V_GAIN = 0.04          # uncertainty-weighting constant: K = K0 * RD^2/(RD^2+V)
RD0 = 0.35             # initial rating deviation (logit scale)
RD_MIN = 0.06
C_DECAY = 0.010        # RD^2 growth per day of inactivity (= c^2 in manuscript)
K_ITEM = 0.02          # difficulty drift rate

# Expected-score weights: E_hat = sigma(w_d * theta_d + w_s * theta_s - w_dj * d_j - w_dc * d_c)
W_THETA_D = 0.5     # discipline-level ability weight
W_THETA_S = 0.5     # skill-level ability weight
W_DIFF_SKILL = 0.7  # skill difficulty weight
W_DIFF_TTYPE = 0.3  # test-type difficulty weight

# Theta update multipliers (asymmetric: skill adapts fastest)
W_UPDATE_GLOBAL = 0.5   # theta_g update multiplier
W_UPDATE_DISC = 1.0     # theta_d update multiplier
W_UPDATE_SKILL = 1.5    # theta_s update multiplier
W_DIFF_UPDATE = 1.0     # diff_skill update multiplier
W_DIFF_TT_UPDATE = 0.5  # diff_ttype update multiplier

XGB_POINT = dict(n_estimators=1000, learning_rate=0.04, max_depth=10,
                 subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                 reg_lambda=1.5, reg_alpha=0.1, objective="reg:squarederror",
                 tree_method="hist", n_jobs=-1, random_state=SEED)
XGB_Q = {k: v for k, v in XGB_POINT.items() if k not in ("objective", "reg_alpha")}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------------------
# LAYER 1 — streaming rating state
# ----------------------------------------------------------------------
def build_rating_features(df: pd.DataFrame) -> pd.DataFrame:
    """Single chronological pass. Feature values for each row are the state
    BEFORE that attempt is processed — leakage-free by construction."""
    df = df.sort_values(COL["ts"], kind="mergesort")
    rid = df.index.to_numpy()
    df = df.reset_index(drop=True)
    y = (df[COL["score"]] / df[COL["maxscore"]]).clip(0, 1).to_numpy()
    ts = pd.to_datetime(df[COL["ts"]]).astype("int64").to_numpy() / 86.4e12  # days

    students = df[COL["student"]].to_numpy()
    discs = df[COL["disc"]].astype(str).to_numpy()
    skills = df[COL["skill"]].astype(str).to_numpy()
    ttypes = df[COL["ttype"]].astype(str).to_numpy()

    theta_g, theta_d, theta_s = {}, {}, {}        # abilities
    rd_g, rd_d = {}, {}                            # rating deviations (variance)
    diff_skill, diff_ttype = {}, {}                # difficulties
    last_seen, n_g, n_d, n_s = {}, {}, {}, {}
    surprise = {}                                  # recent (s - E) history

    n = len(df)
    out = np.zeros((n, 15), dtype=np.float64)

    for t in range(n):
        i, s_key, sk = students[t], (students[t], discs[t]), (students[t], skills[t])
        j, c = skills[t], ttypes[t]

        th_g = theta_g.get(i, 0.0)
        th_d = theta_d.get(s_key, th_g)
        th_s = theta_s.get(sk, th_d)
        dj = diff_skill.get(j, 0.0)
        dc = diff_ttype.get(c, 0.0)

        # inactivity decay of RD (variance grows with days since last attempt)
        gap = ts[t] - last_seen.get(i, ts[t])
        v_g = min(rd_g.get(i, RD0**2) + C_DECAY * gap, RD0**2)
        v_d = min(rd_d.get(s_key, RD0**2) + C_DECAY * gap, RD0**2)

        e_hat = sigmoid(W_THETA_D * th_d + W_THETA_S * th_s
                        - W_DIFF_SKILL * dj - W_DIFF_TTYPE * dc)
        sur = surprise.get(s_key, [])

        out[t] = [th_g, th_d, th_s,
                  np.sqrt(v_g), np.sqrt(v_d),
                  dj, dc, e_hat,
                  np.mean(sur[-5:]) if sur else 0.0,
                  min(gap, 365.0),
                  n_g.get(i, 0), n_d.get(s_key, 0), n_s.get(sk, 0),
                  0.0, 0.0]  # last 2 slots reserved for calendar (filled below)

        # ---- update state with the observed outcome ----
        err = y[t] - e_hat
        k_g = K0 * v_g / (v_g + V_GAIN)
        k_d = K0 * v_d / (v_d + V_GAIN)
        theta_g[i] = th_g + W_UPDATE_GLOBAL * k_g * err
        theta_d[s_key] = th_d + W_UPDATE_DISC * k_d * err
        theta_s[sk] = th_s + W_UPDATE_SKILL * k_d * err
        diff_skill[j] = dj - W_DIFF_UPDATE * K_ITEM * err
        diff_ttype[c] = dc - W_DIFF_TT_UPDATE * K_ITEM * err
        # posterior-style shrinkage: each observation reduces the variance
        rd_g[i] = max(v_g * V_GAIN / (v_g + V_GAIN), RD_MIN**2)
        rd_d[s_key] = max(v_d * V_GAIN / (v_d + V_GAIN), RD_MIN**2)
        last_seen[i] = ts[t]
        n_g[i] = n_g.get(i, 0) + 1
        n_d[s_key] = n_d.get(s_key, 0) + 1
        n_s[sk] = n_s.get(sk, 0) + 1
        surprise[s_key] = (sur + [err])[-5:]

    cols = ["theta_g", "theta_disc", "theta_skill", "rd_g", "rd_disc",
            "diff_skill", "diff_ttype", "e_hat", "surprise5",
            "days_gap", "n_global", "n_disc", "n_skill", "_r1", "_r2"]
    feats = pd.DataFrame(out, columns=cols, index=df.index).drop(columns=["_r1", "_r2"])

    dt = pd.to_datetime(df[COL["ts"]])
    feats["month"], feats["dayofweek"] = dt.dt.month.values, dt.dt.dayofweek.values
    for src, dst in [("disc", "disc_code"), ("skill", "skill_code"), ("ttype", "ttype_code")]:
        feats[dst] = df[COL[src]].astype("category").cat.codes.values
    feats["y"] = y * 100.0
    feats["ts"] = dt.values
    feats["rid"] = rid
    return feats


RATING_FEATURES = ["theta_g", "theta_disc", "theta_skill", "rd_g", "rd_disc",
                   "diff_skill", "diff_ttype", "e_hat", "surprise5", "days_gap",
                   "n_global", "n_disc", "n_skill", "month", "dayofweek",
                   "disc_code", "skill_code", "ttype_code"]

# Combined: rating + aggregate supplement + derived features (26 total)
# NOTE: e_hat_x100 (= 100*e_hat) removed — perfect linear transform of
# e_hat, contributes no new information.
GRACE_FEATURES = RATING_FEATURES + [
    "roll3", "exp_avg", "lag1", "test_num",
    "theta_range", "score_momentum", "gap_bucket", "is_cold",
]

RF_PARAMS = dict(
    n_estimators=300, max_depth=20, min_samples_leaf=25,
    max_features="sqrt", n_jobs=-1, random_state=SEED,
)


# ----------------------------------------------------------------------
# Clean aggregate baseline (leakage-free version of the current pipeline)
# ----------------------------------------------------------------------
def build_aggregate_features(df: pd.DataFrame, leaky: bool = False) -> pd.DataFrame:
    """leaky=True reproduces the notebook's flawed constructions for E2:
    rolling/expanding WITHOUT shift(1) plus bfill. leaky=False is correct."""
    df = df.sort_values([COL["student"], COL["disc"], COL["ts"]]).reset_index()
    y = (df[COL["score"]] / df[COL["maxscore"]]).clip(0, 1) * 100
    df["_y"] = y
    g = df.groupby([COL["student"], COL["disc"]])["_y"]

    if leaky:
        df["roll3"] = g.transform(lambda s: s.rolling(3, min_periods=1).mean().bfill().ffill())
        df["exp_avg"] = g.transform(lambda s: s.expanding().mean()).bfill().ffill()
        df["lag1"] = g.transform(lambda s: s.shift(1).bfill().ffill())
    else:
        df["roll3"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        df["exp_avg"] = g.transform(lambda s: s.shift(1).expanding().mean())
        df["lag1"] = g.transform(lambda s: s.shift(1))

    dt = pd.to_datetime(df[COL["ts"]])
    df["month"], df["dayofweek"] = dt.dt.month, dt.dt.dayofweek
    df["test_num"] = df.groupby([COL["student"], COL["disc"]]).cumcount()
    for src, dst in [("disc", "disc_code"), ("skill", "skill_code"), ("ttype", "ttype_code")]:
        df[dst] = df[COL[src]].astype("category").cat.codes
    out = df[["roll3", "exp_avg", "lag1", "month", "dayofweek", "test_num",
              "disc_code", "skill_code", "ttype_code", "_y", "index"]].copy()
    out = out.rename(columns={"_y": "y", "index": "rid"})
    out["ts"] = dt.values
    return out.sort_values("ts").reset_index(drop=True)


AGG_FEATURES = ["roll3", "exp_avg", "lag1", "month", "dayofweek", "test_num",
                "disc_code", "skill_code", "ttype_code"]


def build_combined_features(feats, agg):
    """Merge rating features with aggregate supplement and add derived features."""
    combined = feats.merge(
        agg[["rid", "roll3", "exp_avg", "lag1", "test_num"]],
        on="rid", how="left"
    )
    combined["e_hat_x100"] = combined["e_hat"] * 100
    combined["theta_range"] = combined["theta_g"] - combined["theta_disc"]
    combined["score_momentum"] = combined["surprise5"] * combined["n_disc"].clip(upper=20)
    combined["gap_bucket"] = pd.cut(combined["days_gap"],
                                     bins=[-1, 0, 1, 3, 7, 14, 30, 365],
                                     labels=False).fillna(6).astype(int)
    combined["is_cold"] = (combined["n_disc"] < 3).astype(int)
    return combined


# ----------------------------------------------------------------------
# LAYERS 2 + 3 — boosted quantile core + Mondrian CQR
# ----------------------------------------------------------------------
def chrono_split(feats, train_frac=0.7, cal_frac=0.1):
    n = len(feats)
    i1, i2 = int(n * train_frac), int(n * (train_frac + cal_frac))
    return feats.iloc[:i1], feats.iloc[i1:i2], feats.iloc[i2:]


def fit_grace(train, features):
    Xtr, ytr = train[features], train["y"]
    point = XGBRegressor(**XGB_POINT).fit(Xtr, ytr)
    qlo = XGBRegressor(objective="reg:quantileerror", quantile_alpha=ALPHA / 2,
                       **XGB_Q).fit(Xtr, ytr)
    qhi = XGBRegressor(objective="reg:quantileerror", quantile_alpha=1 - ALPHA / 2,
                       **XGB_Q).fit(Xtr, ytr)
    return point, qlo, qhi


def rd_strata(rd, edges):
    return np.digitize(rd, edges)


def conformalize(cal, test, models, features, strat_col="rd_disc"):
    """Mondrian split-CQR stratified by rating-deviation terciles."""
    _, qlo, qhi = models
    edges = np.quantile(cal[strat_col], [1 / 3, 2 / 3])
    s_cal = rd_strata(cal[strat_col].to_numpy(), edges)
    s_te = rd_strata(test[strat_col].to_numpy(), edges)

    lo_c, hi_c = qlo.predict(cal[features]), qhi.predict(cal[features])
    scores = np.maximum(lo_c - cal["y"].to_numpy(), cal["y"].to_numpy() - hi_c)

    lo_t, hi_t = qlo.predict(test[features]), qhi.predict(test[features])
    qhat = np.zeros(len(test))
    for s in np.unique(s_te):
        sc = scores[s_cal == s]
        k = int(np.ceil((len(sc) + 1) * (1 - ALPHA)))
        qhat[s_te == s] = np.sort(sc)[min(k, len(sc)) - 1]
    return np.clip(lo_t - qhat, 0, 100), np.clip(hi_t + qhat, 0, 100), s_te


def evaluate(test, models, lo, hi, s_te, features, label=""):
    point = models[0]
    yhat = point.predict(test[features])
    y = test["y"].to_numpy()
    print(f"\n=== {label} ===")
    print(f"R2={r2_score(y, yhat):.4f}  MAE={mean_absolute_error(y, yhat):.3f}")
    cover = (y >= lo) & (y <= hi)
    print(f"coverage(marginal)={cover.mean():.3f}  mean width={(hi - lo).mean():.2f}")
    for s in np.unique(s_te):
        m = s_te == s
        print(f"  RD-stratum {s}: n={m.sum():>8}  coverage={cover[m].mean():.3f}"
              f"  width={(hi - lo)[m].mean():.2f}")
    return dict(r2=r2_score(y, yhat), mae=mean_absolute_error(y, yhat),
                coverage=cover.mean(), width=(hi - lo).mean())


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------
def load(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=[COL["score"], COL["maxscore"], COL["ts"]])
    df = df[pd.to_numeric(df[COL["maxscore"]], errors="coerce") > 0]
    df[COL["ts"]] = pd.to_datetime(df[COL["ts"]], errors="coerce")
    return df.dropna(subset=[COL["ts"]])


def e1_e3(df):
    """Clean baseline vs GRACE combined features (E1 + E3)."""
    agg = build_aggregate_features(df, leaky=False).dropna(subset=["exp_avg"])
    tr, cal, te = chrono_split(agg)
    m = fit_grace(tr, AGG_FEATURES)
    yhat = m[0].predict(te[AGG_FEATURES])
    print(f"\n[E1] Clean aggregates: R2={r2_score(te['y'], yhat):.4f} "
          f"MAE={mean_absolute_error(te['y'], yhat):.3f}")

    feats = build_rating_features(df)
    feats = feats[feats["n_global"] > 0]
    feats = build_combined_features(feats, agg)
    tr, cal, te = chrono_split(feats)
    models = fit_grace(tr, GRACE_FEATURES)
    lo, hi, s_te = conformalize(cal, te, models, GRACE_FEATURES)
    evaluate(te, models, lo, hi, s_te, GRACE_FEATURES, "[E3] GRACE")

    # history-depth strata comparison (cold-start analysis)
    for lab, mask in [("<5", te["n_disc"] < 5), ("5-10", te["n_disc"].between(5, 10)),
                      (">10", te["n_disc"] > 10)]:
        sub = te[mask]
        if len(sub) == 0:
            continue
        yh = models[0].predict(sub[GRACE_FEATURES].fillna(-999))
        print(f"  history {lab}: n={len(sub):>8} R2={r2_score(sub['y'], yh):.4f} "
              f"MAE={mean_absolute_error(sub['y'], yh):.3f}")


def leakage_audit(df):
    """E2: quantify inflation from leaky features x random split."""
    from sklearn.model_selection import train_test_split
    results = {}
    for leaky in (False, True):
        feats = build_aggregate_features(df, leaky=leaky).dropna(subset=["exp_avg"])
        for split in ("chrono", "random"):
            if split == "chrono":
                tr, _, te = chrono_split(feats, 0.8, 0.0)
            else:
                tr, te = train_test_split(feats, test_size=0.2, random_state=SEED)
            m = XGBRegressor(**XGB_POINT).fit(tr[AGG_FEATURES], tr["y"])
            yhat = m.predict(te[AGG_FEATURES])
            r2 = r2_score(te["y"], yhat)
            results[(leaky, split)] = r2
            print(f"leaky={leaky!s:5}  split={split:6}  "
                  f"R2={r2:.4f}  MAE={mean_absolute_error(te['y'], yhat):.3f}")
    inflation = results[(True, "random")] - results[(False, "chrono")]
    print(f"\nTotal optimism (leaky+random vs clean+chrono): dR2 = {inflation:+.4f}")


def rolling(df):
    """E5: rolling-origin evaluation of GRACE (2018-2020->2021 ... ->2025H1)."""
    feats = build_rating_features(df)
    feats = feats[feats["n_global"] > 0]
    agg = build_aggregate_features(df, leaky=False).dropna(subset=["exp_avg"])
    feats = build_combined_features(feats, agg)
    yrs = feats["ts"].dt.year
    folds = [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
    rows = []
    for tr_end, te_yr in folds:
        tr = feats[(yrs >= 2018) & (yrs <= tr_end)]
        te = feats[yrs == te_yr]
        if te_yr == 2025:
            te = te[te["ts"] < "2025-07-01"]
        if len(tr) < 1000 or len(te) < 1000:
            continue
        cut = int(len(tr) * 0.85)          # last 15% of train window -> calibration
        trn, cal = tr.iloc[:cut], tr.iloc[cut:]
        models = fit_grace(trn, GRACE_FEATURES)
        lo, hi, s_te = conformalize(cal, te, models, GRACE_FEATURES)
        r = evaluate(te, models, lo, hi, s_te, GRACE_FEATURES,
                     f"train 2018-{tr_end} -> test {te_yr}")
        rows.append({"train_end": tr_end, "test": te_yr,
                     "n_train": len(tr), "n_test": len(te), **r})
    out = pd.DataFrame(rows)
    out.to_csv("grace_rolling_origin.csv", index=False)
    print("\n", out[["r2", "mae", "coverage", "width"]].agg(["mean", "std"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--experiment", default="e1_e3",
                    choices=["e1_e3", "leakage_audit", "rolling"])
    args = ap.parse_args()
    data = load(args.data)
    {"e1_e3": e1_e3, "leakage_audit": leakage_audit, "rolling": rolling}[args.experiment](data)
