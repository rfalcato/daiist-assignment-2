"""
Cost model shared by train.py (to pick the operating point) and app.py (to
recompute it live when the sliders move). No training, no model loading here.

Unit of decision: one drone photo of one insulator string.
Score of a photo: highest confidence the model gives to any FAULT box
(defect, brokeninsulator, brokenpart). No fault box -> score 0.

Three outcomes per photo, given two thresholds  t_low <= t_high:
    score <  t_low            -> PASS     (insulator left in service)
    t_low <= score < t_high   -> REVIEW   (an engineer looks at the photo)
    score >= t_high           -> DISPATCH (repair crew sent to the tower)

A two-outcome model ("model decides alone") is the special case t_low == t_high.

Rates are measured on the test split *conditional on the true class*, then
re-weighted with the assumed production base rate, because the test split has
far more faulty insulators than a real line does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
COSTS_FILE = ROOT / "costs.json"
GRID = np.round(np.linspace(0.0, 1.0, 101), 2)


def load_costs(path: Path = COSTS_FILE) -> dict:
    raw = json.loads(Path(path).read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


# --------------------------------------------------------------------------- #
# outcome shares
# --------------------------------------------------------------------------- #
def outcome_shares(scores: np.ndarray, t_low: float, t_high: float) -> tuple[float, float, float]:
    """Share of `scores` falling in PASS / REVIEW / DISPATCH."""
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0.0
    t_high = max(t_high, t_low)
    passed = float(np.mean(scores < t_low))
    alert = float(np.mean(scores >= t_high))
    return passed, 1.0 - passed - alert, alert


def three_way_counts(scores: np.ndarray, y: np.ndarray, t_low: float, t_high: float) -> pd.DataFrame:
    """Raw test-split counts: rows = truth, columns = outcome."""
    t_high = max(t_high, t_low)
    out = np.where(scores < t_low, "PASS", np.where(scores >= t_high, "DISPATCH", "REVIEW"))
    rows = []
    for label, mask in (("faulty", y == 1), ("healthy", y == 0)):
        rows.append({
            "truth": label,
            "PASS": int(np.sum(out[mask] == "PASS")),
            "REVIEW": int(np.sum(out[mask] == "REVIEW")),
            "DISPATCH": int(np.sum(out[mask] == "DISPATCH")),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# cost
# --------------------------------------------------------------------------- #
def review_cost_per_case(c: dict) -> float:
    return c["review_seconds"] / 3600.0 * c["reviewer_hourly_eur"]


def expected_cost(scores: np.ndarray, y: np.ndarray, c: dict, t_low: float, t_high: float) -> dict:
    """Expected euros per case at the assumed base rate, plus review load."""
    pi = c["base_rate"]
    pos, neg = scores[y == 1], scores[y == 0]
    p_pass, p_rev, p_alert = outcome_shares(pos, t_low, t_high)
    n_pass, n_rev, n_alert = outcome_shares(neg, t_low, t_high)
    cr = review_cost_per_case(c)

    # The reviewer is assumed to be right: a reviewed faulty case becomes a
    # correct alert, a reviewed healthy case becomes a correct pass.
    cost_pos = p_pass * c["cost_fn_eur"] + p_rev * (cr + c["cost_tp_eur"]) + p_alert * c["cost_tp_eur"]
    cost_neg = n_pass * c["cost_tn_eur"] + n_rev * (cr + c["cost_tn_eur"]) + n_alert * c["cost_fp_eur"]
    per_case = pi * cost_pos + (1 - pi) * cost_neg

    review_share = pi * p_rev + (1 - pi) * n_rev
    reviews_day = review_share * c["cases_per_day"]
    hours_day = reviews_day * c["review_seconds"] / 3600.0
    people = hours_day / c["reviewer_productive_hours"]
    return {
        "t_low": float(t_low),
        "t_high": float(max(t_high, t_low)),
        "cost_per_case": per_case,
        "cost_per_1000": per_case * 1000,
        "cost_per_day": per_case * c["cases_per_day"],
        "review_share": review_share,
        "reviews_per_day": reviews_day,
        "review_hours_per_day": hours_day,
        "reviewers_needed": people,
        "recall_faulty": p_rev + p_alert,  # caught by alert or by a person
        "missed_faulty_rate": p_pass,
        "false_alert_rate": n_alert,
        "alert_share": pi * p_alert + (1 - pi) * n_alert,
    }


def sweep_single(scores: np.ndarray, y: np.ndarray, c: dict, grid=GRID) -> pd.DataFrame:
    """Model decides alone: one threshold, alert if score >= t."""
    rows = []
    pos, neg = scores[y == 1], scores[y == 0]
    for t in grid:
        r = expected_cost(scores, y, c, t, t)
        tp = int(np.sum(pos >= t)); fp = int(np.sum(neg >= t))
        r["threshold"] = float(t)
        r["test_precision"] = tp / (tp + fp) if (tp + fp) else float("nan")
        r["test_recall"] = tp / len(pos) if len(pos) else float("nan")
        rows.append(r)
    return pd.DataFrame(rows)


def optimise_band(scores: np.ndarray, y: np.ndarray, c: dict, grid=GRID,
                  max_reviewers: float | None = None) -> dict:
    """Best (t_low, t_high) on the grid, optionally capped by reviewer head-count."""
    best = None
    for i, tl in enumerate(grid):
        for th in grid[i:]:
            r = expected_cost(scores, y, c, tl, th)
            if max_reviewers is not None and r["reviewers_needed"] > max_reviewers + 1e-9:
                continue
            if best is None or r["cost_per_case"] < best["cost_per_case"] - 1e-12:
                best = r
    return best


def reference_strategies(scores: np.ndarray, y: np.ndarray, c: dict, band: dict, single: dict) -> pd.DataFrame:
    """The four ways of running the site that the brief asks to compare."""
    pi, F = c["base_rate"], c["cases_per_day"]
    cr = review_cost_per_case(c)
    nothing = pi * c["cost_fn_eur"] + (1 - pi) * c["cost_tn_eur"]
    human_all = cr + pi * c["cost_tp_eur"] + (1 - pi) * c["cost_tn_eur"]
    human_people = F * c["review_seconds"] / 3600.0 / c["reviewer_productive_hours"]
    rows = [
        {"strategy": "Do nothing (no inspection model)", "cost_per_1000": nothing * 1000,
         "reviewers_needed": 0.0, "review_share": 0.0},
        {"strategy": "Engineer reviews every photo", "cost_per_1000": human_all * 1000,
         "reviewers_needed": human_people, "review_share": 1.0},
        {"strategy": f"Model alone (t={single['t_low']:.2f})", "cost_per_1000": single["cost_per_1000"],
         "reviewers_needed": 0.0, "review_share": 0.0},
        {"strategy": f"Model + review band [{band['t_low']:.2f}, {band['t_high']:.2f})",
         "cost_per_1000": band["cost_per_1000"], "reviewers_needed": band["reviewers_needed"],
         "review_share": band["review_share"]},
    ]
    df = pd.DataFrame(rows)
    df["cost_per_day"] = df["cost_per_1000"] / 1000 * F
    return df


def roc_auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC (probability a faulty photo scores above a healthy one)."""
    pos, neg = scores[y == 1], scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
