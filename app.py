"""
Assignment 2 - Gradio dashboard (run with `uv run python main.py app`).

Loads only what train.py already produced: weights/best.pt and artifacts/.
Nothing is trained here. Moving a slider recomputes the cost model from the
saved per-image test scores (decision.py), it never re-runs the network on the
test set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from ultralytics import YOLO  # noqa: E402

import decision  # noqa: E402

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
WEIGHTS = ROOT / "weights" / "best.pt"

model = YOLO(str(WEIGHTS))
NAMES = model.names
COSTS = decision.load_costs()
POS = COSTS["positive_classes"]
POS_IDX = [k for k, v in NAMES.items() if v in POS]

SCORES = pd.read_csv(ART / "test_scores.csv")
Y = SCORES["y"].to_numpy()
S = SCORES["score"].to_numpy()
METRICS = json.loads((ART / "metrics.json").read_text())
OP = json.loads((ART / "operating_point.json").read_text())
STATS = json.loads((ART / "dataset_stats.json").read_text())
BASE_DEC = pd.read_csv(ART / "baselines_decision.csv")
T_SINGLE = OP["single_threshold"]["t_low"]
T_LOW0, T_HIGH0 = OP["band_chosen"]["t_low"], OP["band_chosen"]["t_high"]

COST_KEYS = ["base_rate", "cost_fn_eur", "cost_fp_eur", "cost_tp_eur", "cases_per_day",
             "review_seconds", "reviewer_hourly_eur", "reviewer_productive_hours",
             "reviewers_available"]


def costs_from(*vals) -> dict:
    c = dict(COSTS)
    c.update(dict(zip(COST_KEYS, [float(v) for v in vals])))
    return c


# --------------------------------------------------------------------------- #
# image tab
# --------------------------------------------------------------------------- #
def predict(img, t_low, t_high):
    if img is None:
        return None, "Upload an image."
    t_high = max(t_high, t_low)
    r = model.predict(img, conf=min(t_low, 0.99) if t_low > 0 else 0.01, verbose=False)[0]
    cls = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) else np.array([], int)
    conf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.array([])
    faults = conf[np.isin(cls, POS_IDX)]
    score = float(faults.max()) if len(faults) else 0.0
    if score >= t_high:
        verdict = f"🔴 DISPATCH repair crew (fault score {score:.2f} ≥ {t_high:.2f})"
    elif score >= t_low:
        verdict = f"🟠 ENGINEER REVIEW (fault score {score:.2f} in [{t_low:.2f}, {t_high:.2f}))"
    else:
        verdict = f"🟢 PASS - leave in service (fault score {score:.2f} < {t_low:.2f})"
    counts = {NAMES[int(k)]: int((r.boxes.cls == k).sum()) for k in r.boxes.cls.unique()} if len(r.boxes) else {}
    annotated = r.plot()[:, :, ::-1]  # BGR -> RGB
    return annotated, f"**{verdict}**\n\nBoxes shown (conf ≥ {t_low:.2f}): {counts or 'none'}"


# --------------------------------------------------------------------------- #
# operating-point tab
# --------------------------------------------------------------------------- #
def cost_curve(c, t_low, t_high):
    sw = decision.sweep_single(S, Y, c)
    best = sw.loc[sw["cost_per_case"].idxmin()]
    band = decision.expected_cost(S, Y, c, t_low, t_high)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sw["threshold"], sw["cost_per_1000"], label="model alone: cost per 1,000 cases")
    ax.axvline(T_SINGLE, ls=":", c="grey", label=f"chosen in training t={T_SINGLE:.2f}")
    ax.scatter([best["threshold"]], [best["cost_per_1000"]], c="red", zorder=5,
               label=f"minimum now t={best['threshold']:.2f} (€{best['cost_per_1000']:.0f})")
    ax.axvspan(t_low, max(t_high, t_low), color="orange", alpha=0.2,
               label=f"review band → €{band['cost_per_1000']:.0f}/1,000")
    ax.set_xlabel("confidence threshold on max fault score")
    ax.set_ylabel("expected € per 1,000 cases")
    ax.set_ylim(0, min(sw["cost_per_1000"].max(), 3 * max(best["cost_per_1000"], 1)) * 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def update(t_low, t_high, *cost_vals):
    c = costs_from(*cost_vals)
    t_high = max(t_high, t_low)
    r = decision.expected_cost(S, Y, c, t_low, t_high)
    counts = decision.three_way_counts(S, Y, t_low, t_high)
    single = decision.expected_cost(S, Y, c, T_SINGLE, T_SINGLE)
    strat = decision.reference_strategies(S, Y, c, r, single).round(2)
    staff_ok = "✅ fits" if r["reviewers_needed"] <= c["reviewers_available"] else "⚠️ exceeds"
    summary = (
        f"### €{r['cost_per_1000']:.0f} per 1,000 cases  ·  €{r['cost_per_day']:.0f} per day\n"
        f"- Sent to engineer review: **{r['review_share']:.1%}** of cases "
        f"→ {r['reviews_per_day']:.0f}/day → {r['review_hours_per_day']:.1f} h "
        f"→ **{r['reviewers_needed']:.2f} reviewers** ({staff_ok} {c['reviewers_available']:.0f} available)\n"
        f"- Auto-dispatches: {r['alert_share']:.1%} of cases · faults caught: "
        f"{r['recall_faulty']:.1%} · missed: {r['missed_faulty_rate']:.1%}\n"
        f"- Base rate assumed: {c['base_rate']:.1%} (test split: {Y.mean():.1%})"
    )
    return counts, summary, cost_curve(c, t_low, t_high), strat


def per_class_table():
    return pd.DataFrame(METRICS["per_class"]).round(3)


def detection_baselines():
    return pd.DataFrame(METRICS["baselines_detection"]).round(3)


def cm_table():
    cm = METRICS["confusion_matrix"]
    labels = cm["labels"]
    df = pd.DataFrame(cm["matrix"], index=[f"pred {l}" for l in labels],
                      columns=[f"true {l}" for l in labels])
    return df.reset_index().rename(columns={"index": ""})


def stats_table():
    rows = []
    for split, s in STATS["splits"].items():
        for cls in STATS["classes"]:
            sz = s["object_size"].get(cls, {})
            rows.append({"split": split, "images": s["images"], "class": cls,
                         "instances": s["instances"][cls],
                         "images_with_class": s["images_with_class"][cls],
                         "median_box_%_of_image": round(sz.get("median_area_pct", 0), 3),
                         "share_boxes_<1%": round(sz.get("share_below_1pct", 0), 3)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
with gr.Blocks(title="Insulator fault triage") as demo:
    gr.Markdown(
        "# Power-line insulator fault triage\n"
        f"Positive = any fault: **{', '.join(POS)}**. One photo = one insulator string from a drone pass. "
        "Each photo gets PASS / ENGINEER REVIEW / DISPATCH from its highest fault-class confidence. "
        f"Data: `{STATS['source']}` · test images: {len(Y)}."
    )
    with gr.Row():
        t_low = gr.Slider(0, 1, value=T_LOW0, step=0.01, label="t_low: below → PASS")
        t_high = gr.Slider(0, 1, value=T_HIGH0, step=0.01, label="t_high: at/above → DISPATCH")

    with gr.Accordion("Cost assumptions (from costs.json; edit to test sensitivity)", open=False):
        with gr.Row():
            cost_inputs = [gr.Number(value=COSTS[k], label=k) for k in COST_KEYS]

    with gr.Tab("Operating point"):
        with gr.Row():
            with gr.Column():
                summary = gr.Markdown()
                gr.Markdown("**Test-split confusion matrix at these thresholds** (counts; a person resolves REVIEW)")
                counts = gr.Dataframe(interactive=False)
            curve = gr.Plot()
        gr.Markdown("**Against the alternatives** (same costs, same base rate)")
        strat = gr.Dataframe(interactive=False)

    with gr.Tab("Try an image"):
        with gr.Row():
            img_in = gr.Image(type="numpy", label="Upload an insulator photo")
            img_out = gr.Image(label="Predictions at t_low")
        verdict = gr.Markdown()
        gr.Button("Run").click(predict, [img_in, t_low, t_high], [img_out, verdict])
        img_in.change(predict, [img_in, t_low, t_high], [img_out, verdict])

    with gr.Tab("Per-class results"):
        o = METRICS["overall"]
        gr.Markdown(f"**Overall (test):** AP50 {o['AP50']:.3f} · mAP50-95 {o['mAP50-95']:.3f} · "
                    f"P {o['precision']:.3f} · R {o['recall']:.3f}")
        gr.Dataframe(per_class_table(), label="Per class, test split", interactive=False)
        gr.Dataframe(cm_table(), label="Detection confusion matrix (test)", interactive=False)
        gr.Markdown("**Baselines, detection metrics (same test split)**")
        gr.Dataframe(detection_baselines(), interactive=False)
        gr.Markdown("**Baselines, on the business decision (best threshold for each)**")
        gr.Dataframe(BASE_DEC.round(3), interactive=False)
        cm_png = ART / "test_confusion_matrix.png"
        if cm_png.exists():
            gr.Image(str(cm_png), label="Confusion matrix (Ultralytics)")

    with gr.Tab("Dataset"):
        rs = STATS.get("resplit", {})
        if rs:
            gr.Markdown(f"**Re-split by photo id** (published v1 had no test split): {rs}")
        gr.Dataframe(stats_table(), interactive=False)

    inputs = [t_low, t_high, *cost_inputs]
    outputs = [counts, summary, curve, strat]
    for comp in inputs:
        comp.change(update, inputs, outputs)
    demo.load(update, inputs, outputs)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("APP_PORT", 7860)))
