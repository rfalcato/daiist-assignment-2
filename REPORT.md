# Assignment 2 Report

- **Name**:
- **Student ID**:
- **Email**:
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

> Scaffold only. Every `✍️` is yours to write in your own words (course AI policy).
> Numbers come from `artifacts/` after `uv run python main.py train`.

## Business problem, and the cost of being wrong

- **Decision:** a drone inspects a power line and photographs each insulator string. Each photo is PASS (leave in service), ENGINEER REVIEW, or DISPATCH (send a repair crew to the tower).
- **Who acts:** ✍️
- **False negative (damaged insulator passed):** ✍️ what happens and roughly how many €
- **False positive (healthy insulator, crew sent):** ✍️
- **Why one is worse and by roughly how much:** ✍️ (ratio cost_fn / cost_fp)

## Dataset

- **Roboflow Universe:** workspace `insulators-ljwyo`, project `insulators-c4gwk`, **version 1**
- **Link:** https://universe.roboflow.com/insulators-ljwyo/insulators-c4gwk/dataset/1
- **License:** CC BY 4.0. Downloaded as-is, not forked.
- **Classes:** insulator, defect, brokeninsulator, brokenpart. Positive = any of defect / brokeninsulator / brokenpart.
- **What one image is:** ✍️
- **Why it serves the decision (and where it only resembles it):** ✍️

### Re-split (important)
The published v1 has **958 train / 141 valid / 0 test**, and filenames show pre-augmented copies of the same photo (`160605_JPG`, `160605d_JPG`, `160605v_JPG`…). `train.ipynb` pools all images, groups copies by photo id, and makes a seeded 70/15/15 split by photo, so no photo appears in two splits. Result: `artifacts/dataset_stats.json → resplit`, per-image mapping: `artifacts/split_manifest.csv`.

✍️ Why you did this, and how many photos were leaking across the published train/valid split.

## What the dataset's composition implies

Source: `artifacts/dataset_stats.json` (also shown in the dashboard's Dataset tab).

| split | images | insulator | defect | brokeninsulator | brokenpart |
|---|---|---|---|---|---|
| train | | | | | |
| valid | | | | | |
| test  | | | | | |

Object size (median box % of image): ✍️

✍️ Interpretation. Points to consider: how many test images per class, which numbers are too thin to trust, the share of faulty vs healthy photos compared with a real line, the fact that "copies" are augmentations rather than new photos (so the effective sample size is smaller), and what you therefore cannot conclude.

## Finetuning and baselines

- **Model:** yolo11n detection, finetuned from the COCO checkpoint. Why detection and why n: ✍️
- **Where trained / settings:** see `artifacts/operating_point.json → train_settings` ✍️ (machine, epochs, imgsz, batch, minutes)

| model (same test split) | AP50 | mAP50-95 | image-level AUC | best cost / 1,000 |
|---|---|---|---|---|
| Finetuned yolo11n | | | | |
| COCO yolo11n zero-shot | 0 | 0 | | |
| Trivial: predict nothing | 0 | 0 | 0.5 | |

(`artifacts/metrics.json`, `artifacts/baselines_decision.csv`)

✍️ What changed, and was the gap what you expected?

## Per-class evaluation and failure modes

| class | precision | recall | AP50 | mAP50-95 |
|---|---|---|---|---|
| insulator | | | | |
| defect | | | | |
| brokeninsulator | | | | |
| brokenpart | | | | |

Confusion matrix: `artifacts/test_confusion_matrix.png`

✍️ Which class carries the headline number, which drags it down, and what gets confused with what (especially faults predicted as background or plain insulator, which are your FNs).

## Cost matrix and operating point

| outcome | € | where the figure comes from |
|---|---|---|
| FN | | ✍️ |
| FP | | ✍️ |
| TP | | ✍️ |
| TN | | ✍️ |
| Base rate | | ✍️ |

Sweep: `artifacts/cost_sweep.csv` (curve in the dashboard). Chosen single threshold: ✍️

✍️ The assumption that most changes the answer, and at what value it flips (use the dashboard's cost inputs to find it).

## What a person still has to check

- Thresholds: t_low = ✍️, t_high = ✍️ (`operating_point.json → band_chosen`)
- Share sent to review, reviews/day, minutes each, people needed: ✍️
- Realistic? ✍️

| strategy | € / 1,000 | € / day | reviewers |
|---|---|---|---|
| Do nothing | | | |
| Engineer reviews every photo | | | |
| Model alone | | | |
| Model + review band | | | |

(`artifacts/strategies.csv`)

## Limitations & next steps

✍️

## Generative AI use disclosure

✍️ Example of what to state: which tool, what it was used for (e.g. pipeline and dashboard code, dataset search), and that the cost figures, thresholds and analysis are your own.
