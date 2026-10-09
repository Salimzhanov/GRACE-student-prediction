# GRACE — Data Cleaning Protocol

This document reproduces every exclusion criterion applied to the raw platform
export before any modelling, addressing Reviewer Concern 1.

---

## Raw export

| Field | Value |
|---|---|
| Source | UNT-preparation platform export (anonymized; data shared under data-use agreement) |
| Raw rows | 3,715,413 |
| Columns used | `StudentClientId`, `Discipline`, `SkillName`, `TestTypeCategoryName`, `TestTypeId`, `Score`, `MaxSkillScore`, `DateTime` |

---

## Step-by-step exclusion (sample-flow table)

| Step | Rule | Rows removed | Rows remaining |
|---|---|---|---|
| 0 — raw export | — | — | 3,715,413 |
| 1 — pseudo-skill removal | `SkillName` in the 21-name exclusion list below | 1,445,517 | 2,269,896 |
| 2 — temporal filter | `DateTime` unparseable **or** before 2018-01-01 | 7 | 2,269,889 |
| 3 — numeric coercion | `Score` or `MaxSkillScore` non-numeric or null | included in step 4 | — |
| 4 — range filter | `MaxSkillScore ≤ 0`, `Score < 0`, or `Score > MaxSkillScore` | 42,438 | 2,227,451 |
| 5 — deduplication | exact duplicate rows | 7 | 2,227,444 |
| **Clean corpus** | | **1,487,969 removed (40.0%)** | **2,227,444** |

**Reconciliation with manuscript row count (2,208,313 vs 2,227,444):**
The pipeline applies one additional filter after the steps above: records whose `SkillName` maps to a pseudo-skill code (via category encoding) are dropped during feature construction, removing a further 19,131 rows.  The manuscript reports 2,208,313 cleaned records and 55,482 students, which is the post-pipeline count.  The 2,227,444 figure above is the pre-pipeline cleaning result.

**Reconciliation with Table 2 (no impossible scores):** Table 2's
percentage-score distribution is computed *after* step 4 removes records
where `Score > MaxSkillScore` or `Score < 0`.  Therefore, the
percentage column `y = 100 × Score / MaxSkillScore` is bounded to
[0, 100] by construction and Table 2 correctly reports no out-of-range
values.

---

## Pseudo-skill exclusion list (21 names)

These administrative/metadata fields were identified by two criteria:
(a) they appear exclusively in non-scoring worksheet rows, not in subject
assessments; and (b) they carry no genuine student-ability signal
(e.g., free-text comments, logistic identifiers, redundant raw-point
columns).

| # | SkillName (Kazakh/Russian) | Description |
|---|---|---|
| 1 | `Номер теста` | Test sequence number |
| 2 | `Длительность` | Session duration |
| 3 | `Темы` | Lesson topics (free text) |
| 4 | `Темы среза/дз (прошлый урок)` | Prior-lesson topics |
| 5 | `Количественные характеристики` | Quantitative characteristics |
| 6 | `Ментор` | Mentor name |
| 7 | `Туториал` | Tutorial flag |
| 8 | `Время потока` | Stream/cohort time |
| 9 | `Оценка учителя` | Teacher rating |
| 10 | `Школа` | School identifier |
| 11 | `Отдел Менторов` | Mentor department |
| 12 | `Обратная связь получена` | Feedback-received flag |
| 13 | `КРА` | KRA (administrative code) |
| 14 | `МКЕ` | MKE (administrative code) |
| 15 | `Оценка среза` | Section grade |
| 16 | `Баллы за тест` | Raw test points (redundant with Score) |
| 17 | `Попытки` | Attempt counter |
| 18 | `Ранг` | Rank |
| 19 | `Комментарий` | Comment field |
| 20 | `Комментарий.` | Comment field (variant spelling) |
| 21 | `Срез Общее` | Section overall |

The complete list is embedded in `grace_pipeline.py` as
`PSEUDO_SKILL_NAMES` so that exclusion is applied automatically whenever
`load()` is called, ensuring full reproducibility.

---

## Feature-selection transparency (Concern 3)

All feature and rating-parameter choices were finalised on the
**2018–2020 development window** and **frozen before any test window
(2021–2025) was examined**.  The 19 additional features in the optimised
model (`OPTIMIZED_FEATURES` in `run_grace_optimized.py`) were motivated
by prior diagnostics:

| Group | Feature(s) | Motivation |
|---|---|---|
| A — max\_score encoding | `max_score`, `log_max_score`, `score_granularity` | Mean score varies 18–77 pp across `max_score` values |
| B — finer test type | `ttype_id_code` | 464 unique test-type IDs vs 62 categories |
| C — per-skill stats | `skill_lag1`, `skill_roll3`, `skill_exp_avg`, `skill_n` | Skill-level history differs from discipline-level |
| D — multi-window | `roll5`, `roll10`, `exp_std` | Single 3-window misses medium-term trend |
| E — per-level gaps | `days_gap_disc`, `days_gap_skill`, `gap_bucket_disc` | Discipline and skill inactivity decay at different rates |
| F — interaction | `is_cold_skill`, `e_hat_residual`, `ability_diff_gap`, `trend_5v5`, `roll5_std` | Non-linear interactions between rating state and history |

No test-period outcomes were used to select or tune any feature or
rating parameter.
