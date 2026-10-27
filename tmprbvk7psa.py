import json, os
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")            # draw plots off-screen; Gradio displays them
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

import decision

ROOT = Path.cwd()
ART = ROOT / "artifacts"

model = YOLO(str(ROOT / "weights" / "best.pt"))
NAMES = model.names                                     # {0: 'brokeninsulator', ...}
COSTS = decision.load_costs()
POS = COSTS["positive_classes"]
POS_IDX = [k for k, v in NAMES.items() if v in POS]     # which class ids count as a fault

SCORES = pd.read_csv(ART / "test_scores.csv")           # one row per test photo
Y, S = SCORES["y"].to_numpy(), SCORES["score"].to_numpy()
METRICS = json.loads((ART / "metrics.json").read_text())
OP = json.loads((ART / "operating_point.json").read_text())
STATS = json.loads((ART / "dataset_stats.json").read_text())
BASE_DEC = pd.read_csv(ART / "baselines_decision.csv")

T_SINGLE = OP["single_threshold"]["t_low"]                          # best threshold, model alone
T_LOW0, T_HIGH0 = OP["band_chosen"]["t_low"], OP["band_chosen"]["t_high"]  # chosen review band
COST_KEYS = ["base_rate", "cost_fn_eur", "cost_fp_eur", "cost_tp_eur", "cases_per_day",
             "review_seconds", "reviewer_hourly_eur", "reviewer_productive_hours", "reviewers_available"]
print("loaded", len(Y), "test photos; chosen band", T_LOW0, T_HIGH0)

def predict(img, t_low, t_high):
    if img is None:
        return None, "Upload an image."
    t_high = max(t_high, t_low)
    r = model.predict(img, conf=max(min(t_low, 0.99), 0.01), verbose=False)[0]
    cls = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) else np.array([], int)
    conf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.array([])
    faults = conf[np.isin(cls, POS_IDX)]
    score = float(faults.max()) if len(faults) else 0.0
    if score >= t_high:
        verdict = f"🔴 DISPATCH repair crew (fault score {score:.2f} ≥ {t_high:.2f})"
    elif score >= t_low:
        verdict = f"🟠 ENGINEER REVIEW (fault score {score:.2f} in [{t_low:.2f}, {t_high:.2f}))"
    else:
        verdict = f"🟢 PASS, leave in service (fault score {score:.2f} < {t_low:.2f})"
    counts = {NAMES[int(k)]: int((cls == k).sum()) for k in np.unique(cls)}
    return r.plot()[:, :, ::-1], f"**{verdict}**\n\nBoxes shown: {counts or 'none'}"

def costs_from(*vals):
    c = dict(COSTS)
    c.update(dict(zip(COST_KEYS, [float(v) for v in vals])))
    return c

def cost_curve(c, t_low, t_high):
    sw = decision.sweep_single(S, Y, c)
    best = sw.loc[sw["cost_per_case"].idxmin()]
    band = decision.expected_cost(S, Y, c, t_low, t_high)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sw["threshold"], sw["cost_per_1000"], label="model alone: € per 1,000 photos")
    ax.axvline(T_SINGLE, ls=":", c="grey", label=f"chosen in training t={T_SINGLE:.2f}")
    ax.scatter([best["threshold"]], [best["cost_per_1000"]], c="red", zorder=5,
               label=f"minimum with these costs t={best['threshold']:.2f} (€{best['cost_per_1000']:.0f})")
    ax.axvspan(t_low, max(t_high, t_low), color="orange", alpha=0.2,
               label=f"review band → €{band['cost_per_1000']:.0f} / 1,000")
    ax.set_xlabel("threshold on max fault score"); ax.set_ylabel("expected € per 1,000 photos")
    ax.set_ylim(0, min(sw["cost_per_1000"].max(), 3 * max(best["cost_per_1000"], 1)) * 1.05)
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    return fig

def update(t_low, t_high, *cost_vals):
    c = costs_from(*cost_vals)
    t_high = max(t_high, t_low)
    r = decision.expected_cost(S, Y, c, t_low, t_high)
    counts = decision.three_way_counts(S, Y, t_low, t_high)
    single = decision.expected_cost(S, Y, c, T_SINGLE, T_SINGLE)
    strat = decision.reference_strategies(S, Y, c, r, single).round(2)
    ok = "✅ fits" if r["reviewers_needed"] <= c["reviewers_available"] else "⚠️ exceeds"
    summary = (f"### €{r['cost_per_1000']:.0f} per 1,000 photos · €{r['cost_per_day']:.0f} per day\n"
               f"- Sent to engineer review: **{r['review_share']:.1%}** → {r['reviews_per_day']:.0f}/day → "
               f"{r['review_hours_per_day']:.1f} h → **{r['reviewers_needed']:.2f} reviewers** "
               f"({ok} {c['reviewers_available']:.0f} available)\n"
               f"- Auto-dispatched: {r['alert_share']:.1%} · faults caught: {r['recall_faulty']:.1%} · "
               f"missed: {r['missed_faulty_rate']:.1%}\n"
               f"- Base rate assumed: {c['base_rate']:.1%} (test split: {Y.mean():.1%})")
    return counts, summary, cost_curve(c, t_low, t_high), strat

def cm_table():
    cm = METRICS["confusion_matrix"]
    df = pd.DataFrame(cm["matrix"], index=[f"pred {l}" for l in cm["labels"]],
                      columns=[f"true {l}" for l in cm["labels"]])
    return df.reset_index().rename(columns={"index": ""})

def stats_table():
    rows = []
    for split, s in STATS["splits"].items():
        for c in STATS["classes"]:
            sz = s["object_size"].get(c, {})
            rows.append({"split": split, "images": s["images"], "class": c, "instances": s["instances"][c],
                         "images_with_class": s["images_with_class"][c],
                         "median_box_%_of_image": round(sz.get("median_area_pct", 0), 2)})
    return pd.DataFrame(rows)

with gr.Blocks(title="Insulator fault triage") as demo:
    gr.Markdown("# Power-line insulator fault triage\n"
                f"Fault classes: **{', '.join(POS)}** · one photo = one insulator string · "
                f"data `{STATS['source']}` · {len(Y)} test photos")
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
                gr.Markdown("**Test photos by truth × outcome** (an engineer resolves REVIEW)")
                counts = gr.Dataframe(interactive=False)
            curve = gr.Plot()
        gr.Markdown("**Against the alternatives** (same costs and base rate)")
        strat = gr.Dataframe(interactive=False)

    with gr.Tab("Try an image"):
        with gr.Row():
            img_in = gr.Image(type="numpy", label="Upload an insulator photo")
            img_out = gr.Image(label="Predictions")
        verdict = gr.Markdown()
        gr.Button("Run").click(predict, [img_in, t_low, t_high], [img_out, verdict])
        img_in.change(predict, [img_in, t_low, t_high], [img_out, verdict])

    with gr.Tab("Per-class results"):
        o = METRICS["overall"]
        gr.Markdown(f"**Overall (test):** AP50 {o['AP50']:.3f} · mAP50-95 {o['mAP50-95']:.3f} · "
                    f"P {o['precision']:.3f} · R {o['recall']:.3f}")
        gr.Dataframe(pd.DataFrame(METRICS["per_class"]).round(3), label="Per class, test split", interactive=False)
        gr.Dataframe(cm_table(), label="Detection confusion matrix (rows = predicted, cols = true)", interactive=False)
        gr.Markdown("**Baselines: detection metrics**")
        gr.Dataframe(pd.DataFrame(METRICS["baselines_detection"]).round(3), interactive=False)
        gr.Markdown("**Baselines: on the business decision (best threshold each)**")
        gr.Dataframe(BASE_DEC.round(3), interactive=False)
        if (ART / "test_confusion_matrix.png").exists():
            gr.Image(str(ART / "test_confusion_matrix.png"), label="Confusion matrix (Ultralytics)")

    with gr.Tab("Dataset"):
        gr.Markdown(f"**Re-split by photo id:** {STATS.get('resplit', {})}")
        gr.Dataframe(stats_table(), interactive=False)

    inputs = [t_low, t_high, *cost_inputs]
    outputs = [counts, summary, curve, strat]
    for comp in inputs:
        comp.change(update, inputs, outputs)
    demo.load(update, inputs, outputs)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("APP_PORT", 7860)))
