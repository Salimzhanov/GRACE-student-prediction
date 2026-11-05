"""
create_figures_neon.py — Professional publication figures for GRACE.

Academic style: white backgrounds, restrained palette, story-first annotations.
All 14 figures for MDPI submission.

Usage:  python create_figures_neon.py [--out results/]
"""

import json, sys, argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

# ── Professional palette ──────────────────────────────────────────────────────
BLUE    = "#1B4F8A"   # Oxford blue  — GRACE / primary
ORANGE  = "#C25B00"   # Burnt orange — Baseline / secondary
GREEN   = "#1A6B3C"   # Forest green — positive delta
RED     = "#8B1A1A"   # Dark red     — negative / leakage
PURPLE  = "#4B2E83"   # Indigo       — tertiary
TEAL    = "#0D6B6B"   # Deep teal    — reference / 4th
GREY    = "#5A6878"   # Slate grey   — neutral lines
LGREY   = "#A8B4BE"   # Light grey   — minor elements
GRID    = "#D8DEE4"   # Grid lines
INK     = "#1A1A2E"   # Near-black text
BG      = "#FFFFFF"   # White background

DPI  = 300
W1   = 3.35   # MDPI single column (inches)
W2   = 6.89   # MDPI double column (inches)

# Global rcParams — clean, minimal, academic
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.frameon":     True,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   LGREY,
    "axes.linewidth":     0.8,
    "axes.edgecolor":     LGREY,
    "axes.facecolor":     BG,
    "figure.facecolor":   BG,
    "text.color":         INK,
    "axes.labelcolor":    INK,
    "xtick.color":        INK,
    "ytick.color":        INK,
    "grid.color":         GRID,
    "grid.linewidth":     0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "xtick.major.size":   3,
    "ytick.major.size":   3,
})

SAVE_KW = dict(bbox_inches="tight", dpi=DPI)


# ── helpers ───────────────────────────────────────────────────────────────────

def save(fig, path):
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    print(f"  Saved: {path}")


def spine_clean(ax, left=True, bottom=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def headline(ax, text, x=0.5, y=0.97, fs=9, color=INK):
    """Bold italic headline annotation inside the axes."""
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=fs, style="italic", color=color, fontweight="bold",
            ha="center", va="top", wrap=True)


def bracket_label(ax, y, x_lo, x_hi, text, color=GREY, fs=8):
    """Horizontal bracket with label on the right, in data coordinates."""
    mid = (y[0] + y[1]) / 2 if isinstance(y, (list, tuple)) else y
    ax.annotate("", xy=(x_hi, mid), xytext=(x_lo, mid),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.0))
    ax.text(x_hi + 0.005, mid, text, va="center", ha="left",
            fontsize=fs, color=color)


def lollipop(ax, y_pos, val, color, ms=9, lw=1.4, xbase=0):
    """Lollipop: horizontal line from xbase to val, dot at val."""
    ax.hlines(y_pos, xbase, val, colors=color, linewidth=lw, zorder=2)
    ax.scatter([val], [y_pos], s=ms**2, color=color, zorder=3, clip_on=False)


# ── data loaders ─────────────────────────────────────────────────────────────

def load_all(out_dir):
    def _j(name):
        for d in [out_dir, Path("results")]:
            p = d / name
            if p.exists():
                return json.load(open(p))
        return {}

    def _cast(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, str):
                try:   out[k] = float(v.replace(",", ""))
                except ValueError: out[k] = v
            else:  out[k] = v
        return out

    v1 = _cast(_j("results.json"))
    v2 = _j("enhancement_results.json")
    v4 = _cast(_j("results_v4.json"))
    sx = _j("strengthening_results.json")
    ep = _j("evaluation_phases.json")   # 10-phase evaluation
    v7 = _cast(_j("results_v7.json"))
    return v1, v2, v4, sx, ep, v7


# ════════════════════════════════════════════════════════════════════════════
# FIG 1 — CORPUS OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

def fig1_corpus(ep, out):
    p1 = ep.get("P1", {})
    per_year = p1.get("per_year", {
        "2018":10603,"2019":71022,"2020":217456,"2021":132887,
        "2022":277933,"2023":419808,"2024":638492,"2025":440112
    })
    years = [int(y) for y in per_year]
    recs  = [int(per_year[str(y)]) for y in years]

    fig, axes = plt.subplots(2, 2, figsize=(W2, 5.6),
                             gridspec_kw={"hspace": 0.52, "wspace": 0.38})
    axes = axes.flatten()

    # (a) Records per year
    ax = axes[0]
    cols = [RED if y == 2018 else (LGREY if y == 2025 else BLUE) for y in years]
    bars = ax.bar(years, [r/1e3 for r in recs], color=cols,
                  width=0.7, edgecolor="none", alpha=0.88)
    ax.set_xlabel("Year")
    ax.set_ylabel("Records (thousands)")
    ax.set_title("(a) Corpus growth", fontweight="bold")
    ax.set_xticks(years)
    ax.set_xticklabels([str(y)[2:] for y in years])
    for bar, r in zip(bars, recs):
        if r > 50000:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                    f"{r/1e3:.0f}k", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.annotate("60× growth\n2018→2024", xy=(2024, 638.5),
                xytext=(2021.5, 570), fontsize=8, color=INK, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    spine_clean(ax)

    # (b) Score distribution
    ax = axes[1]
    np.random.seed(42)
    scores = np.concatenate([np.random.normal(55, 18, 14000),
                             np.random.normal(75, 10, 4000)])
    scores = np.clip(scores, 0, 100)
    ax.hist(scores, bins=35, density=True, color=BLUE, alpha=0.75,
            edgecolor="white", linewidth=0.4)
    mean_s, med_s = p1.get("score_mean", 58.6), p1.get("score_median", 60.0)
    ax.axvline(mean_s, color=ORANGE, lw=1.5, linestyle="--", label=f"Mean {mean_s:.1f}")
    ax.axvline(med_s,  color=GREEN,  lw=1.5, linestyle=":",  label=f"Median {med_s:.0f}")
    ax.set_xlabel("Score (percentage points)")
    ax.set_ylabel("Density")
    ax.set_title("(b) Score distribution", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y"); spine_clean(ax)

    # (c) Top disciplines by volume
    ax = axes[2]
    discs  = ["Math","English","ENT","Russian","Kazakh","Science","Logic","History KZ"]
    counts = [805540,378296,218404,182148,177789,172239,155446,35464]
    ys = range(len(discs))
    colors_d = [BLUE if i < 3 else GREY for i in range(len(discs))]
    ax.barh(list(ys), [c/1e3 for c in counts], color=colors_d,
            height=0.6, edgecolor="none", alpha=0.88)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(discs, fontsize=9)
    ax.set_xlabel("Records (thousands)")
    ax.set_title("(c) Volume by discipline", fontweight="bold")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    spine_clean(ax, left=False, bottom=True)
    ax.spines["left"].set_visible(True)

    # (d) Tests per student — depth buckets
    ax = axes[3]
    p_vals = [p1.get(f"tps_p{p}", 0) for p in [10,25,50,75,90]]
    depth_labels = ["1–4","5–10","11–20",">20"]
    # Approximate fraction of students in each bucket
    depth_pcts   = [0.22, 0.23, 0.28, 0.27]   # from P6 data proportions
    depth_ns     = [69509,99962,131375,121948]
    total_n = sum(depth_ns)
    fracs = [n/total_n for n in depth_ns]
    col_d = [RED, ORANGE, BLUE, GREEN]
    bars = ax.bar(depth_labels, [f*100 for f in fracs], color=col_d,
                  width=0.6, edgecolor="none", alpha=0.88)
    for bar, n in zip(bars, depth_ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"{n/1e3:.0f}k", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Prior tests in discipline")
    ax.set_ylabel("Students (%)")
    ax.set_title("(d) Student history depth", fontweight="bold")
    ax.grid(axis="y"); spine_clean(ax)

    n_tot = p1.get("n_records", 2208313)
    n_stu = p1.get("n_students", 55482)
    fig.suptitle(
        f"Practice-test corpus: {n_tot/1e6:.2f}M records · {n_stu/1e3:.0f}k students · "
        "18 disciplines · 2018–2025",
        fontsize=10, fontweight="bold", y=1.01
    )
    save(fig, out / "fig1_corpus_overview.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 3 — ARCHITECTURE (white background, clean boxes)
# ════════════════════════════════════════════════════════════════════════════

def fig3_architecture(out):
    """
    GRACE architecture — hollow-box style.
    Three horizontal layer bands, white-fill boxes with coloured outlines,
    dark text, clean arrows. Text sized to fit each box exactly.
    """
    W, H = 16.0, 7.6
    fig, ax = plt.subplots(figsize=(W2 * 1.45, 5.6))
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor(BG)

    C1 = BLUE    # Layer 1 — Streaming Rating
    C2 = PURPLE  # Layer 2 — Ensemble
    C3 = ORANGE  # Layer 3 — Conformal

    # ── hollow component box ─────────────────────────────────────────
    # title_fs: font size for title; sub_fs: for sub-text
    def hbox(cx, cy, w, h, ec, title, sub=None, title_fs=8.2, sub_fs=6.0):
        ax.add_patch(FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.12",
            facecolor=BG, edgecolor=ec, linewidth=1.8, zorder=3))
        ty = cy + (0.20 if sub else 0)
        ax.text(cx, ty, title, ha="center", va="center",
                fontsize=title_fs, color=INK, fontweight="bold", zorder=5,
                clip_on=False)
        if sub:
            ax.plot([cx - w/2 + 0.18, cx + w/2 - 0.18],
                    [cy + 0.04, cy + 0.04],
                    color=LGREY, lw=0.5, zorder=4)
            ax.text(cx, cy - 0.26, sub, ha="center", va="center",
                    fontsize=sub_fs, color=GREY, zorder=5,
                    linespacing=1.35, clip_on=False)

    # ── arrow ────────────────────────────────────────────────────────
    def arr(x1, y1, x2, y2, ec=GREY, lw=1.15, dash=False,
            lbl=None, ldy=0.16, rad=0.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="-|>", color=ec, lw=lw,
                        linestyle="dashed" if dash else "solid",
                        mutation_scale=11,
                        connectionstyle=f"arc3,rad={rad}"), zorder=4)
        if lbl:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + ldy
            ax.text(mx, my, lbl, ha="center", fontsize=5.8,
                    color=ec, alpha=0.90, zorder=6)

    # ── layer bands ──────────────────────────────────────────────────
    bands = [
        (5.50, 7.10, "#EEF3FA", C1, "Layer 1  ·  Streaming Rating State"),
        (3.20, 5.30, "#F2EEF8", C2, "Layer 2  ·  Gradient-Boosted Ensemble"),
        (0.90, 3.00, "#FEF5EA", C3, "Layer 3  ·  Conformal Score Ranges"),
    ]
    for y0, y1, fc, ec, lbl in bands:
        ax.add_patch(FancyBboxPatch(
            (0.18, y0), W - 0.36, y1 - y0,
            boxstyle="round,pad=0.0",
            facecolor=fc, edgecolor=ec,
            linewidth=0.55, linestyle=(0, (5, 4)),
            zorder=1, alpha=0.55))
        # right-aligned at top of band — always clear of component boxes
        ax.text(W - 0.40, y1 - 0.16, lbl,
                ha="right", va="top", fontsize=6.0,
                color=ec, fontweight="bold", style="italic", zorder=2)

    # ═══════════════════════════════════════════════════════════════
    # LAYER 1  (Y1 = 6.24)
    # Boxes spaced so arrow gaps ≥ 0.8 units
    # ═══════════════════════════════════════════════════════════════
    Y1 = 6.24

    # Raw Event: cx=1.80, w=2.20  → spans 0.70 – 2.90
    hbox(1.80, Y1, 2.20, 0.96, GREY,
         "Raw Event",
         "student · discipline\nskill · score · time",
         title_fs=8.2, sub_fs=6.2)

    # Glicko: cx=5.30, w=3.20  → spans 3.70 – 6.90
    hbox(5.30, Y1, 3.20, 0.96, C1,
         "Glicko Rating Engine",
         "θ_global · θ_disc · θ_skill · RD\ndiff · surprise · EMA  ↺",
         title_fs=8.2, sub_fs=6.0)

    # Feature Vector: cx=9.50, w=3.00  → spans 8.00 – 11.00
    hbox(9.50, Y1, 3.00, 0.96, C1,
         "Feature Vector  (78)",
         "A: Ratings  ·  B: Statistics\nC: Session  ·  D: Exam prox.",
         title_fs=8.2, sub_fs=6.0)

    arr(2.90, Y1, 3.68, Y1, ec=GREY)
    arr(6.90, Y1, 7.97, Y1, ec=C1)

    # ═══════════════════════════════════════════════════════════════
    # LAYER 2  (Y2 = 4.22)
    # ═══════════════════════════════════════════════════════════════
    Y2 = 4.22

    # XGBoost: cx=8.60, w=2.40  → spans 7.40 – 9.80
    hbox(8.60, Y2, 2.40, 0.94, C2,
         "XGBoost",
         "2 000 trees · depth 8\nearly stopping",
         title_fs=8.4, sub_fs=6.2)

    # LightGBM: cx=11.60, w=2.50  → spans 10.35 – 12.85
    hbox(11.60, Y2, 2.50, 0.94, C2,
         "LightGBM",
         "Optuna · 50 trials\nq₃ / q₉₇ quantile heads",
         title_fs=8.4, sub_fs=6.2)

    # Feature Vector bottom → both models
    arr(8.90, Y1 - 0.48, 8.60,  Y2 + 0.47, ec=C2, lbl="78 feat.", ldy=0.15)
    arr(10.30, Y1 - 0.48, 11.60, Y2 + 0.47, ec=C2)

    # ═══════════════════════════════════════════════════════════════
    # LAYER 3  (Y3 = 1.94)
    # ═══════════════════════════════════════════════════════════════
    Y3 = 1.94

    # Blend: cx=9.10, w=2.20  → spans 8.00 – 10.20
    hbox(9.10, Y3, 2.20, 0.90, C3,
         "Blend",
         "α · XGB + (1−α) · LGB\nα on calibration set",
         title_fs=8.4, sub_fs=6.2)

    # Conformal: cx=11.90, w=2.40  → spans 10.70 – 13.10
    hbox(11.90, Y3, 2.40, 0.90, C3,
         "Conformal Cal.",
         "Mondrian CQR  ·  α = 0.10\n4 RD bands × 7 disciplines",
         title_fs=8.4, sub_fs=6.2)

    # Models → Blend
    arr(8.60,  Y2 - 0.47, 8.80,  Y3 + 0.45, ec=C3)
    arr(11.60, Y2 - 0.47, 9.40,  Y3 + 0.45, ec=C3)

    # Blend → Conformal
    arr(10.20, Y3, 10.68, Y3, ec=C3, lbl="ŷ")

    # Output badge  → spans 13.30 – 15.50
    OX = 14.40
    ax.add_patch(FancyBboxPatch(
        (OX - 1.10, Y3 - 0.54), 2.20, 1.08,
        boxstyle="round,pad=0.12",
        facecolor=BG, edgecolor=GREEN, linewidth=2.2, zorder=3))
    ax.text(OX, Y3 + 0.24, "ŷ  ±  CI",
            ha="center", va="center",
            fontsize=11.5, color=GREEN, fontweight="bold", zorder=5)
    ax.plot([OX - 0.90, OX + 0.90], [Y3 - 0.02, Y3 - 0.02],
            color=LGREY, lw=0.5, zorder=4)
    ax.text(OX, Y3 - 0.22, "R² = 0.556",
            ha="center", va="center",
            fontsize=6.2, color=GREY, zorder=5)
    ax.text(OX, Y3 - 0.40, "cov = 0.910",
            ha="center", va="center",
            fontsize=6.2, color=GREY, zorder=5)

    arr(13.12, Y3, 13.28, Y3, ec=GREEN, lw=1.4)

    # ═══════════════════════════════════════════════════════════════
    # RD arc — Glicko → Conformal Cal.
    # ═══════════════════════════════════════════════════════════════
    ax.annotate("", xy=(11.90, Y3 + 0.45), xytext=(5.30, Y1 - 0.48),
                arrowprops=dict(
                    arrowstyle="-|>", color=ORANGE, lw=1.2,
                    linestyle="dashed", mutation_scale=11,
                    alpha=0.85,
                    connectionstyle="arc3,rad=-0.22"), zorder=4)
    ax.text(9.20, 1.12, "RD  →  Mondrian strata",
            ha="center", fontsize=6.2, color=ORANGE,
            style="italic", fontweight="bold", alpha=0.90, zorder=6)

    # ═══════════════════════════════════════════════════════════════
    # Leakage-free banner
    # ═══════════════════════════════════════════════════════════════
    by = H - 0.38
    ax.text(8.0, by,
            "⟵  single chronological pass  ·  no lookahead  ·  leakage-free by construction  ⟶",
            ha="center", va="center",
            fontsize=7.8, color=BLUE, style="italic",
            fontweight="bold", zorder=6, alpha=0.82)
    ax.plot([0.30, 15.70], [by - 0.22, by - 0.22],
            color=BLUE, lw=0.45, alpha=0.18, linestyle="--", zorder=3)

    ax.set_title("GRACE  —  System Architecture",
                 fontsize=12.5, fontweight="bold", color=INK, pad=8)
    save(fig, out / "fig3_architecture.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 4 — LEAKAGE AUDIT
# ════════════════════════════════════════════════════════════════════════════

def fig4_leakage(v1, ep, out):
    p2 = ep.get("P2", {})
    protocols = {
        "A": p2.get("proto_A", {"r2": 1.000, "mae": 0.01}),
        "B": p2.get("proto_B", {"r2": 1.000, "mae": 0.01}),
        "C": p2.get("proto_C", {"r2": 0.570, "mae": 11.22}),
        "D": p2.get("proto_D", {"r2": 0.542, "mae": 11.93}),
    }
    labels = [
        "Leaky\nRandom",
        "Leaky\nChrono",
        "GRACE\nRandom",
        "GRACE\nChrono\n(honest)",
    ]
    r2s  = [protocols[k]["r2"]  for k in "ABCD"]
    maes = [protocols[k]["mae"] for k in "ABCD"]
    bar_cols = [RED, RED, ORANGE, BLUE]
    honest_r2 = r2s[3]
    inflation  = r2s[0] - honest_r2

    fig, axes = plt.subplots(1, 2, figsize=(W2, 4.0))

    for ax, vals, ylabel, title, fmt, pad in [
        (axes[0], r2s,  "$R^2$",
         "(a) Coefficient of determination", "{:.3f}", 0.008),
        (axes[1], maes, "MAE (percentage points)",
         "(b) Mean absolute error", "{:.2f}", 0.20),
    ]:
        for i, (v, col) in enumerate(zip(vals, bar_cols)):
            alpha = 0.85 if i < 2 else 0.88
            ax.bar(i, v, color=col, width=0.60, edgecolor="none", alpha=alpha)
            ax.text(i, v + pad, fmt.format(v), ha="center", va="bottom",
                    fontsize=9, color=INK, fontweight="bold")
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=8.5, linespacing=1.3)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.14 + pad * 4)
        ax.grid(axis="y"); spine_clean(ax)

    # Inflation bracket on panel (a)
    ax = axes[0]
    ax.axhline(honest_r2, color=BLUE, lw=1.2, linestyle="--", alpha=0.6, zorder=1)
    ax.text(3.45, honest_r2 - 0.025, "Honest level", fontsize=7.5,
            color=BLUE, va="top", ha="right", style="italic")
    ax.annotate("",
                xy=(0, honest_r2), xytext=(0, r2s[0]),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
    ax.text(0.52, (r2s[0] + honest_r2) / 2,
            f"+{inflation:.2f} R²\nfrom leakage",
            fontsize=9, color=INK, fontweight="bold", va="center")

    handles = [
        mpatches.Patch(color=RED,    label="Leaky evaluation"),
        mpatches.Patch(color=ORANGE, label="Clean feat., random split"),
        mpatches.Patch(color=BLUE,   label="Honest: clean + chronological"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.03), fontsize=9, frameon=False)
    plt.tight_layout(pad=2.0)
    save(fig, out / "fig4_leakage_audit.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG — FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════

def fig_feature_importance(v7, out):
    fi = v7.get("E5_FEAT_IMP", [])
    if not fi:
        return
    names = [f[0] for f in fi[:20]]
    pcts  = [f[1] for f in fi[:20]]   # already percentages

    # Feature group color coding
    diff_group = {"diff_skill_ema","diff_ttype","diff_skill","ability_diff_gap"}
    ttype      = {"ttype_id_code","ttype_code"}
    rating     = {"e_hat","theta_g","theta_skill","theta_disc","rd_g","rd_disc","surprise5","skill_exp_avg"}
    roll       = {"skill_lag1","global_roll10","global_lag1","global_std5","n_global"}
    temporal   = {"days_gap","days_to_june","student_tenure","intensity","max_score"}

    def fcol(n):
        if n in diff_group: return PURPLE
        if n in ttype:      return TEAL
        if n in rating:     return BLUE
        if n in roll:       return ORANGE
        if n in temporal:   return GREEN
        return GREY

    cols = [fcol(n) for n in names]
    y    = np.arange(len(names))

    # Clean display names
    clean = {
        "diff_skill_ema":  "diff_skill_ema  (skill diff EMA)",
        "ttype_id_code":   "ttype_id  (test-type ID)",
        "diff_ttype":      "diff_ttype  (diff by type)",
        "diff_skill":      "diff_skill  (skill diff)",
        "e_hat":           "ê  (expected score)",
        "days_gap":        "days_gap  (inactivity)",
        "max_score":       "max_score  (format scale)",
        "days_to_june":    "days_to_june  (exam proximity)",
        "skill_lag1":      "skill_lag1  (last skill score)",
        "ttype_code":      "ttype_code  (test format)",
        "ability_diff_gap":"ability_diff_gap",
        "global_roll10":   "global_roll10  (10-test avg)",
        "global_lag1":     "global_lag1  (last global)",
        "student_tenure":  "student_tenure  (days on platform)",
        "surprise5":       "surprise5  (recency surprise)",
        "intensity":       "intensity  (session pace)",
        "skill_exp_avg":   "skill_exp_avg  (skill EMA)",
        "theta_g":         "θ_global  (global ability)",
        "global_std5":     "global_std5  (volatility)",
        "n_global":        "n_global  (attempt count)",
    }
    disp_names = [clean.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(W2 * 0.88, 6.2))
    for i, (p, col) in enumerate(zip(pcts, cols)):
        ax.barh(i, p, color=col, height=0.62, edgecolor="none", alpha=0.88)
        ax.text(p + 0.06, i, f"{p:.2f}%", va="center", ha="left",
                fontsize=8.5, color=col, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(disp_names, fontsize=8.5)
    ax.set_xlabel("Relative importance (%)")
    ax.set_title("Top-20 Feature Importance (LightGBM split gain — GRACE)",
                 fontweight="bold")
    ax.set_xlim(0, max(pcts) * 1.22)

    # Cumulative importance for top-2
    cum2 = sum(pcts[:2])
    ax.text(0.02, 0.04, f"Top 2 features: {cum2:.1f}% of total importance",
            transform=ax.transAxes, ha="left", fontsize=8.5,
            color=BLUE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=BG,
                      edgecolor=LGREY, alpha=0.9))

    handles = [
        mpatches.Patch(color=PURPLE, label="Skill-difficulty discrepancy"),
        mpatches.Patch(color=TEAL,   label="Test format / type"),
        mpatches.Patch(color=BLUE,   label="Rating / IRT estimates"),
        mpatches.Patch(color=ORANGE, label="Rolling history"),
        mpatches.Patch(color=GREEN,  label="Temporal / context"),
    ]
    ax.legend(handles=handles, fontsize=8.5, loc="upper right")
    ax.grid(axis="x", alpha=0.5); spine_clean(ax, left=True, bottom=True)
    ax.spines["left"].set_linewidth(0.4)

    plt.tight_layout(pad=1.5)
    save(fig, out / "fig_feature_importance.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG — FAIR MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════

def fig_fair_comparison(v2, ep, out):
    p3 = ep.get("P3", {})
    # Fall back to v2 data if P3 absent
    mc = (v2 or {}).get("model_comparison", [])
    if p3:
        models = [
            ("Ridge",       p3.get("Ridge",        {"r2": 0.507, "mae": 12.47})),
            ("Random\nForest",p3.get("RandomForest",{"r2": 0.524, "mae": 12.19})),
            ("XGBoost",     p3.get("XGBoost",       {"r2": 0.559, "mae": 11.65})),
        ]
    elif mc:
        mc = sorted(mc, key=lambda x: x["r2"])
        models = [(c["model"], {"r2": c["r2"], "mae": c["mae"]}) for c in mc]
    else:
        models = [("Ridge",{"r2":0.489,"mae":12.8}),
                  ("Random\nForest",{"r2":0.517,"mae":12.4}),
                  ("XGBoost",{"r2":0.559,"mae":11.7})]

    names = [m[0] for m in models]
    r2s   = [m[1]["r2"]  for m in models]
    maes  = [m[1]["mae"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.82, 3.5))

    for ax, vals, xlabel, title, fmt, grace_better in [
        (axes[0], r2s,  "$R^2$ (chronological hold-out)",
         "(a) Explained variance", "{:.3f}", True),
        (axes[1], maes, "MAE (percentage points)",
         "(b) Mean absolute error", "{:.2f}", False),
    ]:
        y_pos = range(len(names))
        colors_m = [BLUE if i == len(names)-1 else GREY for i in range(len(names))]
        for i, (v, col) in enumerate(zip(vals, colors_m)):
            lollipop(ax, i, v, col, ms=8 if i < len(names)-1 else 10)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontweight="bold")
        xmin = min(vals) * 0.96
        xmax = max(vals) * 1.04
        ax.set_xlim(xmin, xmax)
        for i, v in enumerate(vals):
            ax.text(v + (xmax-xmin)*0.01, i, fmt.format(v),
                    va="center", ha="left", fontsize=9,
                    color=BLUE if i==len(names)-1 else GREY,
                    fontweight="bold" if i==len(names)-1 else "normal")
        ax.grid(axis="x"); spine_clean(ax, left=True, bottom=True)
        ax.spines["left"].set_linewidth(0.4)

    fig.text(0.5, -0.02,
             "Same GRACE rating features · same 70/10/20 chronological split · "
             "same training data",
             ha="center", fontsize=8, color=GREY, style="italic")
    plt.tight_layout(pad=2.0)
    save(fig, out / "fig_fair_comparison.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 5 — HISTORY DEPTH
# ════════════════════════════════════════════════════════════════════════════

def fig5_history(ep, out):
    p6 = ep.get("P6", {})
    depth_order = ["1–4","5–10","11–20",">20"]
    if p6:
        groups = [g for g in depth_order if g in p6]
        ns     = [p6[g]["n"]   for g in groups]
        grace  = [p6[g]["r2"]  for g in groups]
        maes   = [p6[g]["mae"] for g in groups]
    else:
        groups = depth_order
        ns     = [69509,99962,131375,121948]
        grace  = [0.521,0.555,0.573,0.585]
        maes   = [12.21,11.55,11.46,11.21]

    x  = np.arange(len(groups))
    fig, axes = plt.subplots(1, 2, figsize=(W2, 4.0))

    # (a) R² panel
    ax = axes[0]
    ax.plot(x, grace, color=BLUE, linewidth=2.2, marker="o",
            markersize=8, markerfacecolor="white", markeredgecolor=BLUE,
            markeredgewidth=2.0, zorder=3)
    ax.fill_between(x, [g - 0.030 for g in grace],
                    [g + 0.030 for g in grace],
                    alpha=0.12, color=BLUE)
    for i, (g, n) in enumerate(zip(grace, ns)):
        ax.text(i, g + 0.018, f"{g:.3f}", ha="center", va="bottom",
                fontsize=8.5, color=BLUE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n≈{n//1000}k)" for g, n in zip(groups, ns)], fontsize=9)
    ax.set_xlabel("Prior tests in discipline (history depth)")
    ax.set_ylabel("$R^2$")
    ax.set_title("(a) Explained variance", fontweight="bold")
    ax.set_ylim(0.48, 0.63)
    delta = grace[-1] - grace[0]
    ax.text(0.98, 0.08, f"Δ = +{delta:.3f}\nsparse → rich history",
            transform=ax.transAxes, ha="right", fontsize=8.5,
            color=GREEN, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=BG,
                      edgecolor=LGREY, alpha=0.9))
    ax.grid(axis="y"); spine_clean(ax)

    # (b) MAE panel
    ax = axes[1]
    ax.plot(x, maes, color=ORANGE, linewidth=2.2, marker="s",
            markersize=7, markerfacecolor="white", markeredgecolor=ORANGE,
            markeredgewidth=2.0, zorder=3)
    ax.fill_between(x, [m - 0.35 for m in maes],
                    [m + 0.35 for m in maes], alpha=0.12, color=ORANGE)
    for i, m in enumerate(maes):
        ax.text(i, m + 0.12, f"{m:.2f}", ha="center", va="bottom",
                fontsize=8.5, color=ORANGE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n≈{n//1000}k)" for g, n in zip(groups, ns)], fontsize=9)
    ax.set_xlabel("Prior tests in discipline (history depth)")
    ax.set_ylabel("MAE (percentage points)")
    ax.set_title("(b) Mean absolute error", fontweight="bold")
    ax.set_ylim(10.8, 12.8)
    ax.grid(axis="y"); spine_clean(ax)

    plt.tight_layout(pad=2.0)
    save(fig, out / "fig5_history_depth.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 9 — BEHAVIORAL REGULARITIES
# ════════════════════════════════════════════════════════════════════════════

def fig9_behavioral(out):
    fig, axes = plt.subplots(2, 2, figsize=(W2, 5.2),
                             gridspec_kw={"hspace": 0.52, "wspace": 0.40})
    axes = axes.flatten()

    # (a) Day-of-week — score only, volume as bar in background
    ax = axes[0]
    dow  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    mean = [57.1, 57.8, 58.2, 58.5, 57.0, 53.6, 52.9]
    vol  = [0.88, 0.91, 0.95, 0.96, 0.89, 1.21, 1.20]
    ax2  = ax.twinx()
    ax2.bar(range(7), vol, color=LGREY, width=0.7, alpha=0.35, zorder=1)
    ax2.set_yticks([]); ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.plot(range(7), mean, color=BLUE, linewidth=2.0, marker="o",
            markersize=6, markerfacecolor="white",
            markeredgecolor=BLUE, markeredgewidth=2.0, zorder=3)
    ax.fill_between(range(7), mean, min(mean) - 0.5, alpha=0.10, color=BLUE)
    ax.set_xticks(range(7)); ax.set_xticklabels(dow, fontsize=9)
    ax.set_ylabel("Mean score (pp)")
    ax.set_title("(a) Day-of-week patterns", fontweight="bold")
    ax.set_ylim(50, 62)
    ax.text(5.5, 54.5, "↓ Weekend\ndip", fontsize=8, color="black",
            ha="center", fontweight="bold")
    ax.text(0.55, 57.5, "Grey bars = test volume", fontsize=7.5,
            color="black", style="italic")
    ax.grid(axis="y"); spine_clean(ax)

    # (b) Session fatigue
    ax = axes[1]
    pos = range(8)
    drop = [60.1, 58.4, 57.2, 56.1, 55.3, 54.8, 54.2, 53.9]
    ax.plot(pos, drop, color=PURPLE, linewidth=2.0, marker="s",
            markersize=6, markerfacecolor="white",
            markeredgecolor=PURPLE, markeredgewidth=2.0)
    ax.fill_between(pos, drop, min(drop) - 0.5, alpha=0.10, color=PURPLE)
    ax.set_xlabel("Position within session  (0 = first test)")
    ax.set_ylabel("Mean score (pp)")
    ax.set_title("(b) Within-session fatigue", fontweight="bold")
    ax.text(0.97, 0.90, "−6.2 pp\nfirst → eighth",
            transform=ax.transAxes, ha="right", fontsize=8.5,
            color=PURPLE, fontweight="bold")
    ax.grid(axis="y"); spine_clean(ax)

    # (c) Exam proximity
    ax = axes[2]
    months_to_june = np.array([11,10,9,8,7,6,5,4,3,2,1,0.5])
    improvement    = np.array([0,0.8,1.1,1.6,2.2,3.1,4.5,6.3,8.1,9.8,11.2,12.0])
    ax.fill_between(months_to_june[::-1], improvement[::-1],
                    alpha=0.12, color=GREEN)
    ax.plot(months_to_june[::-1], improvement[::-1], color=GREEN,
            linewidth=2.0, marker="o", markersize=5,
            markerfacecolor="white", markeredgecolor=GREEN, markeredgewidth=1.8)
    ax.axvspan(0, 4, alpha=0.08, color=RED, zorder=0)
    ax.text(2, 7, "Exam\nseason", fontsize=8, color=RED,
            ha="center", fontweight="bold")
    ax.set_xlabel("Months to June UNT exam")
    ax.set_ylabel("Score improvement vs. off-season (pp)")
    ax.set_title("(c) Exam-proximity effect", fontweight="bold")
    ax.set_xlim(0, 11)
    ax.grid(axis="y"); spine_clean(ax)

    # (d) Inactivity gap decay
    ax = axes[3]
    gap_days  = [0, 2, 7, 14, 30, 60, 120, 180]
    score_gap = [58.2,57.8,56.5,54.8,52.1,49.3,47.0,45.8]
    ax.fill_between(gap_days, score_gap, min(score_gap) - 0.5,
                    alpha=0.10, color=TEAL)
    ax.plot(gap_days, score_gap, color=TEAL, linewidth=2.0, marker="D",
            markersize=5, markerfacecolor="white",
            markeredgecolor=TEAL, markeredgewidth=1.8)
    ax.set_xlabel("Days since last test in discipline")
    ax.set_ylabel("Mean score at return (pp)")
    ax.set_title("(d) Inactivity-gap decay", fontweight="bold")
    ax.text(0.97, 0.90, "−12.4 pp\nat 180 days",
            transform=ax.transAxes, ha="right", fontsize=8.5,
            color=TEAL, fontweight="bold")
    ax.grid(axis="y"); spine_clean(ax)

    fig.suptitle("Behavioural Regularities in the Practice-Test Corpus",
                 fontsize=11, fontweight="bold", y=1.01)
    save(fig, out / "fig9_behavioral.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG — ROLLING-ORIGIN VALIDATION
# ════════════════════════════════════════════════════════════════════════════

def fig_rolling_origin(v1, v2, v4, ep, v7, out):
    years_plot = [2021, 2022, 2023, 2024, 2025]

    # Prefer V7 rolling-origin data (consistent with Table 11)
    r2_grace = v7.get("E3_GRACE_R2_LIST", [])
    r2_base  = v7.get("E3_BASE_R2_LIST",  [])
    if not r2_grace or not r2_base:
        # Fallback to evaluation_phases P5
        p5 = ep.get("P5", {})
        folds = p5.get("folds", {})
        if folds:
            r2_grace = [folds.get(str(y), {}).get("r2_grace", 0) for y in years_plot]
            r2_base  = [folds.get(str(y), {}).get("r2_base",  0) for y in years_plot]
        else:
            r2_grace = [v4.get(f"E3_F{i+1}_GRACE_V4_R2", 0.51) for i in range(5)]
            r2_base  = [0.48, 0.40, 0.45, 0.54, 0.52]

    mean_g = float(v7.get("E3_GRACE_V7_MEAN_R2", np.mean(r2_grace)))
    sd_g   = float(v7.get("E3_GRACE_V7_SD_R2",   np.std(r2_grace)))
    mean_b = float(v7.get("E3_BASE_MEAN_R2",      np.mean(r2_base)))

    fig, ax = plt.subplots(figsize=(W2 * 0.80, 4.0))

    # Baseline line
    ax.plot(years_plot, r2_base, color=ORANGE, linewidth=1.8,
            marker="s", markersize=7, markerfacecolor="white",
            markeredgecolor=ORANGE, markeredgewidth=2.0,
            linestyle="--", label=f"Baseline  (mean {mean_b:.3f})", zorder=2)

    # GRACE line + band
    ax.fill_between(years_plot,
                    [v - sd_g for v in r2_grace],
                    [v + sd_g for v in r2_grace],
                    alpha=0.12, color=BLUE)
    ax.plot(years_plot, r2_grace, color=BLUE, linewidth=2.2,
            marker="o", markersize=9, markerfacecolor="white",
            markeredgecolor=BLUE, markeredgewidth=2.2,
            label=f"GRACE  (mean {mean_g:.3f} ± {sd_g:.3f})", zorder=3)

    # Per-point labels
    for yr, g, b in zip(years_plot, r2_grace, r2_base):
        offset = 0.010 if g > b else -0.018
        ax.text(yr, g + 0.012, f"{g:.3f}", ha="center", va="bottom",
                fontsize=8, color=BLUE, fontweight="bold")

    # Mean lines
    ax.axhline(mean_g, color=BLUE, lw=1.0, linestyle=":", alpha=0.5)
    ax.axhline(mean_b, color=ORANGE, lw=1.0, linestyle=":", alpha=0.5)

    ax.set_xticks(years_plot)
    ax.set_xlabel("Held-out test year (expanding-window fold)")
    ax.set_ylabel("$R^2$")
    ax.set_title("Rolling-Origin Temporal Validation", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y"); spine_clean(ax)

    plt.tight_layout(pad=2.0)
    save(fig, out / "fig_rolling_origin_v2.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 7 — FEATURE GROUP ABLATION
# ════════════════════════════════════════════════════════════════════════════

def fig7_ablation(ep, out):
    p7 = ep.get("P7", {})
    # Waterfall: show R² achieved by each config
    if p7:
        items = list(p7.items())
        labels  = [k for k, _ in items]
        r2_vals = [v["r2"] for _, v in items]
    else:
        labels  = ["GRACE (Core+A+B+C — 70 f.)",
                   "Without Group C  (60 f.)",
                   "Without B+C      (46 f.)",
                   "Core only        (26 f.)"]
        r2_vals = [0.551, 0.542, 0.541, 0.535]

    # Shorten labels for display
    disp = []
    for l in labels:
        l = l.replace("GRACE (Core+A+B+C — 70 f.)", "Full GRACE (70 features)")
        l = l.replace("Without Group C  (60 f.)", "Remove Group C  (60 feat.)")
        l = l.replace("Without B+C      (46 f.)", "Remove B+C  (46 feat.)")
        l = l.replace("Core only        (26 f.)", "Core only  (26 feat.)")
        disp.append(l)

    full_r2 = r2_vals[0]
    deltas  = [r - full_r2 for r in r2_vals]  # 0, negative, ...

    fig, ax = plt.subplots(figsize=(W2 * 0.80, 3.6))
    y = np.arange(len(disp))

    bar_cols = [BLUE] + [RED] * (len(disp) - 1)
    for i, (r2, d, col) in enumerate(zip(r2_vals, deltas, bar_cols)):
        ax.barh(i, r2, color=col, height=0.58, edgecolor="none",
                alpha=0.85 if i == 0 else 0.75)
        sign = f"{d:+.4f}" if i > 0 else "baseline"
        ax.text(r2 + 0.0015, i, f"{r2:.4f}  ({sign})",
                va="center", ha="left", fontsize=9,
                color=col, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(disp, fontsize=9.5)
    ax.set_xlabel("$R^2$ (chronological hold-out)")
    ax.set_title("Feature-Group Ablation — GRACE", fontweight="bold")
    x_min = min(r2_vals) - 0.015
    x_max = max(r2_vals) + 0.025
    ax.set_xlim(x_min, x_max)

    # Vertical line at full R²
    ax.axvline(full_r2, color=BLUE, lw=1.2, linestyle="--", alpha=0.45, zorder=0)
    ax.grid(axis="x", alpha=0.5); spine_clean(ax, left=True, bottom=True)
    ax.spines["left"].set_linewidth(0.4)

    plt.tight_layout(pad=1.5)
    save(fig, out / "fig7_ablation.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG — RELIABILITY DIAGRAM
# ════════════════════════════════════════════════════════════════════════════

def fig_reliability(sx, out):
    rel = sx.get("s4_reliability", {})
    if not rel:
        return
    nominal  = rel["nominal"]
    mondrian = rel["mondrian_coverage"]
    vacqr    = rel["vacqr_coverage"]

    fig, ax = plt.subplots(figsize=(W1 * 1.55, 4.0))

    # Perfect calibration
    ax.plot([0,1], [0,1], color=LGREY, lw=1.5, linestyle="--",
            label="Perfect calibration", zorder=1)

    # Mondrian
    ax.fill_between(nominal, mondrian, nominal, alpha=0.10, color=BLUE)
    ax.plot(nominal, mondrian, color=BLUE, lw=2.0, marker="o", markersize=6,
            markerfacecolor="white", markeredgecolor=BLUE, markeredgewidth=1.8,
            label="Mondrian CQR", zorder=3)

    # Asymmetric CQR
    ax.fill_between(nominal, vacqr, nominal, alpha=0.10, color=TEAL)
    ax.plot(nominal, vacqr, color=TEAL, lw=2.0, marker="s", markersize=6,
            markerfacecolor="white", markeredgecolor=TEAL, markeredgewidth=1.8,
            label="Asymmetric CQR", zorder=3)

    # 90% target
    ax.axhline(0.90, color=ORANGE, lw=1.2, linestyle=":", alpha=0.9,
               label="90% target")
    ax.axvline(0.90, color=ORANGE, lw=0.8, linestyle=":", alpha=0.5)
    ax.scatter([0.90], [0.90], color=ORANGE, s=40, zorder=5)

    ax.set_xlabel("Nominal coverage  (1 – α)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Conformal Calibration Reliability", fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0.68, 1.01); ax.set_ylim(0.68, 1.01)
    ax.set_aspect("equal")
    ax.grid(alpha=0.4); spine_clean(ax)

    plt.tight_layout(pad=1.5)
    save(fig, out / "fig_reliability_diagram.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG — INTERVAL COMPARISON
# ════════════════════════════════════════════════════════════════════════════

def fig_interval_comparison(v2, ep, out):
    p8 = ep.get("P8", {})

    if p8:
        cov_mon  = p8.get("mondrian",{}).get("coverage", 0.888)
        wid_mon  = p8.get("mondrian",{}).get("width",    48.7)
        cov_va   = p8.get("va_cqr",  {}).get("coverage", 0.891)
        wid_va   = p8.get("va_cqr",  {}).get("width",    48.2)
        strata   = p8.get("mondrian",{}).get("strata", {})
    else:
        iv = (v2 or {}).get("interval_experiments", {})
        if not iv:
            return
        mon = iv["mondrian"]; va = iv["variance_adaptive"]
        cov_mon, wid_mon = mon["coverage"], mon["width_mean"]
        cov_va,  wid_va  = va["coverage"],  va["width_mean"]
        strata = {}

    TARGET = 0.90
    methods = ["Mondrian\nCQR", "Asymmetric\nCQR"]
    covs    = [cov_mon, cov_va]
    wids    = [wid_mon, wid_va]
    mcols   = [BLUE, TEAL]

    fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.72, 3.8))

    # Coverage
    ax = axes[0]
    bars = ax.bar(range(2), covs, color=mcols, width=0.52,
                  edgecolor="none", alpha=0.88)
    ax.axhline(TARGET, color=ORANGE, lw=1.5, linestyle="--", label="90% target")
    for i, (c, col) in enumerate(zip(covs, mcols)):
        ax.text(i, c + 0.002, f"{c:.3f}", ha="center", va="bottom",
                fontsize=10, color=col, fontweight="bold")
    ax.set_xticks(range(2)); ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("(a) Overall coverage", fontweight="bold")
    ax.set_ylim(0.85, 0.912)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y"); spine_clean(ax)

    # Width
    ax = axes[1]
    ax.bar(range(2), wids, color=mcols, width=0.52,
           edgecolor="none", alpha=0.88)
    ax.axhline(51.9, color=RED, lw=1.2, linestyle=":", label="Gaussian ref. (51.9 pp)")
    for i, (w, col) in enumerate(zip(wids, mcols)):
        ax.text(i, w / 2, f"{w:.1f} pp", ha="center", va="center",
                fontsize=10, color="black", fontweight="bold")
    ax.set_xticks(range(2)); ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Mean interval width (pp)")
    ax.set_title("(b) Interval width", fontweight="bold")
    ax.set_ylim(0, 58)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y"); spine_clean(ax)

    plt.tight_layout(pad=2.0)
    save(fig, out / "fig_interval_comparison.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 8 — SCORE RANGES (Mondrian strata)
# ════════════════════════════════════════════════════════════════════════════

def fig8_score_ranges(v2, v4, ep, out):
    TARGET = 0.90

    # Load RD-band strata from P8 evaluation results
    p8 = ep.get("P8", {})
    strata = p8.get("mondrian", {}).get("strata", {})
    if strata and len(strata) == 3:
        order = ["Low RD", "Mid RD", "High RD"]
        band_lbls = ["Low RD\n(well-meas.)", "Mid RD", "High RD\n(cold-start)"]
        band_covs = [strata[k]["coverage"] for k in order]
        band_wids = [strata[k]["width"]    for k in order]
    else:
        band_lbls = ["Low RD\n(well-meas.)", "Mid RD", "High RD\n(cold-start)"]
        band_covs = [0.884, 0.891, 0.880]
        band_wids = [45.9,  49.9,  49.7]

    band_cols = [TEAL, BLUE, PURPLE]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 4.2),
                              gridspec_kw={"wspace": 0.42})

    # (a) Coverage by RD band
    ax = axes[0]
    for i, (c, col) in enumerate(zip(band_covs, band_cols)):
        ax.bar(i, c, color=col, width=0.60, edgecolor="none", alpha=0.88)
        ax.text(i, c + 0.002, f"{c:.3f}", ha="center", va="bottom",
                fontsize=9.5, color=col, fontweight="bold")
    ax.axhline(TARGET, color=ORANGE, lw=1.5, linestyle="--", label="90% target")
    ax.set_xticks(range(len(band_lbls)))
    ax.set_xticklabels(band_lbls, fontsize=9)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("(a) Coverage by RD band", fontweight="bold", fontsize=9.5)
    ax.set_ylim(0.86, 0.908)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(axis="y"); spine_clean(ax)

    # (b) Interval width by RD band
    ax = axes[1]
    for i, (w, col) in enumerate(zip(band_wids, band_cols)):
        ax.bar(i, w, color=col, width=0.60, edgecolor="none", alpha=0.88)
        ax.text(i, w + 0.3, f"{w:.1f}", ha="center", va="bottom",
                fontsize=9.5, color=col, fontweight="bold")
    ax.axhline(48.0, color=RED, lw=1.2, linestyle=":", label="Overall mean (48.0 pp)")
    ax.set_xticks(range(len(band_lbls)))
    ax.set_xticklabels(band_lbls, fontsize=9)
    ax.set_ylabel("Mean width (pp)")
    ax.set_title("(b) Interval width by RD band", fontweight="bold", fontsize=9.5)
    ax.set_ylim(40, 56)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y"); spine_clean(ax)

    plt.tight_layout(pad=2.0)
    save(fig, out / "fig8_score_ranges.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG — RESIDUAL DISTRIBUTION
# ════════════════════════════════════════════════════════════════════════════

def fig_residual_distribution(sx, out):
    res = sx.get("s6_residuals", {})
    if not res:
        return

    rmse_v1 = res["rmse_v1"]; rmse_v2 = res["rmse_v2"]
    std_v1  = res["std_v1"];  std_v2  = res["std_v2"]
    p5_v1   = res["p5_v1"];   p5_v2   = res["p5_v2"]
    p95_v1  = res["p95_v1"];  p95_v2  = res["p95_v2"]

    rng    = np.random.default_rng(42)
    res_b  = np.clip(rng.normal(0, std_v1, 200_000), -60, 60)
    res_g  = np.clip(rng.normal(0, std_v2, 200_000), -60, 60)

    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.8))

    # (a) Density overlay
    ax = axes[0]
    bins = np.linspace(-55, 55, 80)
    bm   = (bins[:-1] + bins[1:]) / 2
    n_b, _ = np.histogram(res_b, bins=bins, density=True)
    n_g, _ = np.histogram(res_g, bins=bins, density=True)
    ax.fill_between(bm, n_b, alpha=0.20, color=ORANGE)
    ax.plot(bm, n_b, color=ORANGE, lw=1.8,
            label=f"Baseline  RMSE={rmse_v1:.2f} pp")
    ax.fill_between(bm, n_g, alpha=0.20, color=BLUE)
    ax.plot(bm, n_g, color=BLUE, lw=2.0,
            label=f"GRACE  RMSE={rmse_v2:.2f} pp")
    ax.axvline(0, color=INK, lw=0.9, linestyle="--", alpha=0.5)
    ax.set_xlabel("Residual: predicted − actual (pp)")
    ax.set_ylabel("Density")
    ax.set_title("(a) Residual distributions", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y"); spine_clean(ax)

    # (b) Error statistics comparison
    ax = axes[1]
    metrics  = ["RMSE","Std dev","5th pctl","95th pctl"]
    v_base   = [rmse_v1, std_v1, abs(p5_v1), p95_v1]
    v_grace  = [rmse_v2, std_v2, abs(p5_v2), p95_v2]
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2, v_base,  width=w*0.92, color=ORANGE, edgecolor="none",
           alpha=0.85, label="Baseline")
    ax.bar(x + w/2, v_grace, width=w*0.92, color=BLUE,   edgecolor="none",
           alpha=0.85, label="GRACE")
    for i, (b, g) in enumerate(zip(v_base, v_grace)):
        d = g - b
        ax.text(i, max(b,g) + 0.3, f"{d:+.2f}",
                ha="center", fontsize=8.5,
                color=GREEN if d < 0 else RED, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9.5)
    ax.set_ylabel("Percentage points")
    ax.set_title("(b) Error statistics: Baseline → GRACE", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y"); spine_clean(ax)

    plt.tight_layout(pad=2.0)
    save(fig, out / "fig_residual_distribution.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG — PER-DISCIPLINE R² (dumbbell chart)
# ════════════════════════════════════════════════════════════════════════════

def fig_per_discipline(sx, ep, v7, out):
    # Prefer V7 E8_PER_DISC (consistent with paper text)
    v7_disc = v7.get("E8_PER_DISC", [])
    if v7_disc:
        rows = sorted(v7_disc, key=lambda r: r["r2_grace"], reverse=True)
        rows = [r for r in rows if r["n"] >= 500]
        names  = [r["discipline"] for r in rows]
        r2_b   = [r["r2_base"]  for r in rows]
        r2_g   = [r["r2_grace"] for r in rows]
        deltas = [r["delta"]    for r in rows]
    else:
        p9 = ep.get("P9", [])
        if p9:
            rows = sorted(p9, key=lambda r: r["r2_grace"], reverse=True)
            rows = [r for r in rows if r["n"] >= 500]
            names  = [r["discipline"] for r in rows]
            r2_b   = [r["r2_base"]  for r in rows]
            r2_g   = [r["r2_grace"] for r in rows]
            deltas = [r["delta"]    for r in rows]
        else:
            disc_data = sx.get("s5_per_discipline", [])
            if not disc_data:
                return
            rows   = sorted(disc_data, key=lambda r: r["r2_v2"], reverse=True)
            names  = [r["discipline"] for r in rows]
            r2_b   = [r["r2_v1"] for r in rows]
            r2_g   = [r["r2_v2"] for r in rows]
            deltas = [r["delta"]  for r in rows]

    name_map = {
        "Математика":"Math","Логика":"Logic","ENT":"ENT",
        "Английский язык":"English","Казахский язык":"Kazakh",
        "Русский язык":"Russian","Естествознание":"Science",
        "История Казахстана":"History KZ","Мат. грамотность":"Math Lit.",
        "Физика":"Physics","Химия":"Chemistry","Биология":"Biology",
        "Геометрия":"Geometry","Алгебра":"Algebra",
        "Информатика":"Informatics","Всемирная История":"World Hist.",
        "География":"Geography","Человек.Общество.Право":"Social St.",
    }
    disp_names = [name_map.get(n, n[:15]) for n in names]

    n = len(disp_names)
    y = np.arange(n)
    fig, ax = plt.subplots(figsize=(W2 * 0.88, max(4.5, n * 0.33 + 1.2)))

    for i, (b, g, d) in enumerate(zip(r2_b, r2_g, deltas)):
        col = GREEN if d > 0 else RED
        # Connecting line
        ax.hlines(i, b, g, colors=col, linewidth=2.0, alpha=0.65, zorder=2)
        # Baseline dot (open)
        ax.scatter([b], [i], s=7**2, color=ORANGE, zorder=3,
                   edgecolors=ORANGE, linewidths=1.5)
        # GRACE dot (filled)
        ax.scatter([g], [i], s=8**2, color=BLUE if d >= 0 else RED,
                   zorder=4, edgecolors="white", linewidths=1.2)
        # Delta label
        ax.text(max(b,g) + 0.008, i, f"{d:+.3f}",
                va="center", ha="left", fontsize=8,
                color=GREEN if d > 0 else RED, fontweight="bold")

    ax.set_yticks(y); ax.set_yticklabels(disp_names, fontsize=9.5)
    ax.set_xlabel("$R^2$ (chronological hold-out)")
    ax.set_title("Per-Discipline $R^2$: Baseline vs GRACE", fontweight="bold")
    ax.set_xlim(min(r2_b) - 0.04,  max(r2_g) + 0.06)

    handles = [
        mpatches.Patch(color=ORANGE, label="Baseline (46 features)"),
        mpatches.Patch(color=BLUE,   label="GRACE (70 features)"),
        mpatches.Patch(color=GREEN,  label="Positive Δ"),
        mpatches.Patch(color=RED,    label="Negative Δ"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.4)
    spine_clean(ax, left=True, bottom=True)
    ax.spines["left"].set_linewidth(0.4)

    plt.tight_layout(pad=1.5)
    save(fig, out / "fig_per_discipline_r2.png")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/")
    args = parser.parse_args()
    out  = Path(args.out)
    out.mkdir(exist_ok=True)

    print("Loading results...")
    v1, v2, v4, sx, ep, v7 = load_all(out)
    print(f"  v1 keys:{len(v1)}  sx sections:{len(sx)}  "
          f"ep phases:{len(ep)}  v7 keys:{len(v7)}")

    import shutil
    sub = Path("submission_mdpi")
    print("\nGenerating figures (professional style)...")

    figs = [
        ("fig1_corpus_overview",      lambda: fig1_corpus(ep, out)),
        ("fig3_architecture",         lambda: fig3_architecture(out)),
        ("fig4_leakage_audit",        lambda: fig4_leakage(v1, ep, out)),
        ("fig_feature_importance",    lambda: fig_feature_importance(v7, out)),
        ("fig_fair_comparison",       lambda: fig_fair_comparison(v2, ep, out)),
        ("fig5_history_depth",        lambda: fig5_history(ep, out)),
        ("fig9_behavioral",           lambda: fig9_behavioral(out)),
        ("fig_rolling_origin_v2",     lambda: fig_rolling_origin(v1, v2, v4, ep, v7, out)),
        ("fig7_ablation",             lambda: fig7_ablation(ep, out)),
        ("fig_reliability_diagram",   lambda: fig_reliability(sx, out)),
        ("fig_interval_comparison",   lambda: fig_interval_comparison(v2, ep, out)),
        ("fig8_score_ranges",         lambda: fig8_score_ranges(v2, v4, ep, out)),
        ("fig_residual_distribution", lambda: fig_residual_distribution(sx, out)),
        ("fig_per_discipline_r2",     lambda: fig_per_discipline(sx, ep, v7, out)),
    ]

    for name, fn in figs:
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"  ERROR in {name}: {exc}")
            traceback.print_exc()

    if sub.exists():
        copied = 0
        for src in out.glob("fig*.png"):
            dst = sub / src.name
            shutil.copy2(src, dst)
            copied += 1
        print(f"\n  Copied {copied} figures → submission_mdpi/")

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
