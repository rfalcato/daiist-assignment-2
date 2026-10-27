"""
Assignment 2 - training pipeline (run with `uv run python main.py train`).

Decision supported: a drone photographs every insulator string on a power
line. For each photo: PASS (insulator fine), send to an engineer for REVIEW,
or DISPATCH a repair crew to replace a damaged insulator.

Stages, all automatic:
 1. download Roboflow Universe dataset insulators-ljwyo/insulators-c4gwk v1
    (1,099 images; classes insulator / defect / brokeninsulator / brokenpart).
    The published version has no test split and its pre-augmented copies of
    one photo straddle train/valid, so it is re-split 70/15/15 by photo id
    (seeded, reproducible) -> artifacts/split_manifest.csv
 2. describe the dataset from its labels  -> artifacts/dataset_stats.json
 3. finetune yolo11n from the COCO checkpoint -> weights/best.pt
 4. evaluate on the TEST split: per-class P/R/AP50/mAP50-95 + confusion matrix
 5. baselines on the same test split: COCO yolo11n zero-shot, and a trivial
    "predict nothing" model
 6. score every test image (max fault confidence) -> artifacts/test_scores.csv
 7. sweep the threshold, cost it with costs.json, choose the operating point
    -> artifacts/cost_sweep.csv, artifacts/operating_point.json

Training knobs come from environment variables so the same script runs on a
CPU Codespace or a GPU (defaults are sized for a 4-core CPU Codespace):
    EPOCHS=20  IMGSZ=512  BATCH=16  FRACTION=1.0  MODEL=yolo11n.pt  DEVICE=auto
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

import decision
from api_key import load_api_key

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "datasets" / "insulators-v1-raw"     # as downloaded
DATA_DIR = ROOT / "datasets" / "insulators-v1-split"   # grouped re-split
ART = ROOT / "artifacts"
WEIGHTS = ROOT / "weights"
RUNS = ROOT / "runs"

WORKSPACE, PROJECT, VERSION = "insulators-ljwyo", "insulators-c4gwk", 1

EPOCHS = int(os.environ.get("EPOCHS", 20))
IMGSZ = int(os.environ.get("IMGSZ", 512))
BATCH = int(os.environ.get("BATCH", 16))
FRACTION = float(os.environ.get("FRACTION", 1.0))
MODEL = os.environ.get("MODEL", "yolo11n.pt")
SEED = 0
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_INFO: dict = {}


def pick_device() -> str:
    d = os.environ.get("DEVICE", "auto")
    if d != "auto":
        return d
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- #
# 1. data
# --------------------------------------------------------------------------- #
def download() -> Path:
    """Download the public version as-is into RAW_DIR, then re-split locally."""
    yaml_path = RAW_DIR / "data.yaml"
    if not yaml_path.exists():
        from roboflow import Roboflow
        rf = Roboflow(api_key=load_api_key())
        rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
            "yolov11", location=str(RAW_DIR), overwrite=True)
    return resplit(yaml.safe_load(yaml_path.read_text()))


def group_key(name: str) -> str:
    """Original photo id. The dataset ships pre-augmented copies of the same
    photo with suffixes (e.g. 160605_JPG, 160605d_JPG, 160605v_JPG, 160605h_JPG),
    and Roboflow appends '.rf.<hash>'. All copies must land in the same split."""
    stem = name.split(".rf.")[0]
    stem = re.sub(r"(_(jpe?g|png))+$", "", stem, flags=re.I)  # e.g. 000000795h_jpeg_jpg
    return re.sub(r"-?[hvd]$", "", stem, flags=re.I)


def resplit(cfg: dict) -> Path:
    """The published v1 has train/valid only (no test) and lets augmented
    copies of one photo sit in both. Pool everything, group copies by photo id,
    and make a seeded 70/15/15 grouped split. Deterministic: same data, same split."""
    pool = []
    for split in ("train", "valid", "test"):
        d = RAW_DIR / split / "images"
        if d.exists():
            pool += [(p, split) for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXT]
    groups = sorted({group_key(p.name) for p, _ in pool})
    rng = np.random.default_rng(SEED)
    rng.shuffle(groups)
    n = len(groups)
    cut1, cut2 = int(0.70 * n), int(0.85 * n)
    assign = {g: ("train" if i < cut1 else "valid" if i < cut2 else "test")
              for i, g in enumerate(groups)}

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    rows = []
    for p, orig in pool:
        g = group_key(p.name)
        new = assign[g]
        for sub, src in (("images", p), ("labels", p.parent.parent / "labels" / (p.stem + ".txt"))):
            dst = DATA_DIR / new / sub
            dst.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy(src, dst / src.name)
        rows.append({"image": p.name, "photo_id": g, "original_split": orig, "new_split": new})
    man = pd.DataFrame(rows)
    ART.mkdir(exist_ok=True)
    man.to_csv(ART / "split_manifest.csv", index=False)

    # How bad was the published split? photos whose copies sat in both train and valid.
    leak = man.groupby("photo_id")["original_split"].nunique()
    SPLIT_INFO.update({
        "published_split": man["original_split"].value_counts().to_dict(),
        "new_split_images": man["new_split"].value_counts().to_dict(),
        "unique_photos": int(n),
        "images_per_photo_mean": round(len(man) / n, 2),
        "photos_leaking_across_published_train_valid": int((leak > 1).sum()),
    })
    print("[split]", SPLIT_INFO)

    names = cfg["names"]
    fixed = DATA_DIR / "data_abs.yaml"
    fixed.write_text(yaml.safe_dump({"path": str(DATA_DIR), "train": "train/images",
                                     "val": "valid/images", "test": "test/images",
                                     "nc": len(names), "names": names}, sort_keys=False))
    return fixed


def class_names(data_yaml: Path) -> list[str]:
    names = yaml.safe_load(data_yaml.read_text())["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def images_in(split: str) -> list[Path]:
    d = DATA_DIR / split / "images"
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.exists() else []


def label_for(img: Path) -> Path:
    return img.parent.parent / "labels" / (img.stem + ".txt")


def read_labels(img: Path) -> np.ndarray:
    """rows of (cls, x, y, w, h), normalised."""
    p = label_for(img)
    if not p.exists() or p.stat().st_size == 0:
        return np.zeros((0, 5))
    rows = [list(map(float, l.split()[:5])) for l in p.read_text().splitlines() if l.strip()]
    return np.array(rows) if rows else np.zeros((0, 5))


# --------------------------------------------------------------------------- #
# 2. describe the dataset
# --------------------------------------------------------------------------- #
def describe(names: list[str]) -> dict:
    stats = {"source": f"{WORKSPACE}/{PROJECT}/{VERSION}",
             "url": f"https://universe.roboflow.com/{WORKSPACE}/{PROJECT}/dataset/{VERSION}",
             "classes": names, "resplit": SPLIT_INFO, "splits": {}}
    for split in ("train", "valid", "test"):
        imgs = images_in(split)
        inst = {n: 0 for n in names}
        imgs_with = {n: 0 for n in names}
        areas = {n: [] for n in names}
        empty = 0
        for img in imgs:
            lab = read_labels(img)
            if len(lab) == 0:
                empty += 1
            for k, n in enumerate(names):
                sel = lab[lab[:, 0] == k] if len(lab) else lab
                inst[n] += len(sel)
                imgs_with[n] += int(len(sel) > 0)
                areas[n] += list(sel[:, 3] * sel[:, 4]) if len(sel) else []
        size = {}
        for n in names:
            a = np.array(areas[n])
            size[n] = ({"median_area_pct": float(np.median(a) * 100),
                        "p10_area_pct": float(np.percentile(a, 10) * 100),
                        "p90_area_pct": float(np.percentile(a, 90) * 100),
                        "share_below_1pct": float(np.mean(a < 0.01)),
                        "median_side_px_at_imgsz": float(np.median(np.sqrt(a)) * IMGSZ)}
                       if len(a) else {})
        stats["splits"][split] = {"images": len(imgs), "background_images": empty,
                                  "instances": inst, "images_with_class": imgs_with,
                                  "object_size": size}
    return stats


# --------------------------------------------------------------------------- #
# 3-4. train + evaluate
# --------------------------------------------------------------------------- #
def train(data_yaml: Path, device: str) -> Path:
    model = YOLO(MODEL)
    model.train(data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                fraction=FRACTION, device=device, seed=SEED, deterministic=True,
                workers=2, patience=50, project=str(RUNS), name="finetune",
                exist_ok=True, plots=True, verbose=False)
    best = RUNS / "finetune" / "weights" / "best.pt"
    WEIGHTS.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS / "best.pt")
    for f in ("results.csv", "results.png"):
        if (RUNS / "finetune" / f).exists():
            shutil.copy(RUNS / "finetune" / f, ART / f"train_{f}")
    return WEIGHTS / "best.pt"


def evaluate(weights: Path, data_yaml: Path, names: list[str], device: str) -> dict:
    m = YOLO(str(weights)).val(data=str(data_yaml), split="test", imgsz=IMGSZ,
                                batch=BATCH, device=device, project=str(RUNS),
                                name="test_eval", exist_ok=True, plots=True, verbose=False)
    box = m.box
    per_class = []
    for i, n in enumerate(names):
        idx = list(box.ap_class_index).index(i) if i in list(box.ap_class_index) else None
        per_class.append({
            "class": n,
            "precision": float(box.p[idx]) if idx is not None else 0.0,
            "recall": float(box.r[idx]) if idx is not None else 0.0,
            "AP50": float(box.ap50[idx]) if idx is not None else 0.0,
            "mAP50-95": float(box.ap[idx]) if idx is not None else 0.0,
        })
    cm = m.confusion_matrix.matrix.astype(int).tolist()
    for f in ("confusion_matrix.png", "confusion_matrix_normalized.png", "PR_curve.png"):
        src = RUNS / "test_eval" / f
        if src.exists():
            shutil.copy(src, ART / f"test_{f}")
    return {"overall": {"precision": float(box.mp), "recall": float(box.mr),
                        "AP50": float(box.map50), "mAP50-95": float(box.map)},
            "per_class": per_class,
            "confusion_matrix": {"labels": names + ["background"], "matrix": cm,
                                 "orientation": "rows = predicted, cols = true (Ultralytics)"}}


# --------------------------------------------------------------------------- #
# 5-6. per-image scores for the finetuned model and the COCO baseline
# --------------------------------------------------------------------------- #
def max_conf_per_class(model: YOLO, imgs: list[Path], device: str, n_cls: int) -> np.ndarray:
    out = np.zeros((len(imgs), n_cls))
    for start in range(0, len(imgs), 64):
        chunk = [str(p) for p in imgs[start:start + 64]]
        for j, r in enumerate(model.predict(chunk, conf=0.001, imgsz=IMGSZ, device=device,
                                            verbose=False, stream=True)):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            for c in range(n_cls):
                if np.any(cls == c):
                    out[start + j, c] = conf[cls == c].max()
    return out


def score_test(weights: Path, names: list[str], pos_idx: list[int], device: str) -> pd.DataFrame:
    imgs = images_in("test")
    labels = [read_labels(p) for p in imgs]
    df = pd.DataFrame({"image": [p.name for p in imgs]})
    for k, n in enumerate(names):
        df[f"gt_{n}"] = [int(np.sum(l[:, 0] == k)) if len(l) else 0 for l in labels]
    df["y"] = (df[[f"gt_{names[k]}" for k in pos_idx]].sum(axis=1) > 0).astype(int)

    ft = max_conf_per_class(YOLO(str(weights)), imgs, device, len(names))
    for k, n in enumerate(names):
        df[f"score_{n}"] = ft[:, k]
    df["score"] = df[[f"score_{names[k]}" for k in pos_idx]].max(axis=1)

    # COCO zero-shot: COCO has no insulator/fault classes. The only signal it can give
    # this decision is "how confident is it that *anything* is in the photo",
    # i.e. the max confidence over all 80 COCO classes.
    coco = YOLO("yolo11n.pt")
    df["coco_any_score"] = max_conf_per_class(coco, imgs, device, 80).max(axis=1)
    # Trivial baseline: predict nothing -> score 0 everywhere -> never alerts.
    df["trivial_score"] = 0.0
    return df


# --------------------------------------------------------------------------- #
# 7. cost sweep + operating point
# --------------------------------------------------------------------------- #
def choose_operating_point(df: pd.DataFrame, c: dict) -> dict:
    y = df["y"].to_numpy()
    s = df["score"].to_numpy()
    sweep = decision.sweep_single(s, y, c)
    sweep.to_csv(ART / "cost_sweep.csv", index=False)
    single = sweep.loc[sweep["cost_per_case"].idxmin()].to_dict()
    single = decision.expected_cost(s, y, c, single["threshold"], single["threshold"])

    band_free = decision.optimise_band(s, y, c)
    band = decision.optimise_band(s, y, c, max_reviewers=c["reviewers_available"])
    strategies = decision.reference_strategies(s, y, c, band, single)
    strategies.to_csv(ART / "strategies.csv", index=False)

    # Baselines judged on the same decision.
    base = []
    for name, col in (("Finetuned yolo11n (max fault score)", "score"),
                      ("COCO yolo11n zero-shot (max any-class score)", "coco_any_score"),
                      ("Trivial: predict nothing", "trivial_score")):
        sc = df[col].to_numpy()
        sw = decision.sweep_single(sc, y, c)
        b = sw.loc[sw["cost_per_case"].idxmin()]
        base.append({"model": name, "image_level_AUC": decision.roc_auc(sc, y),
                     "best_threshold": float(b["threshold"]),
                     "best_cost_per_1000": float(b["cost_per_1000"]),
                     "recall_at_best": float(b["test_recall"]),
                     "precision_at_best": float(b["test_precision"])})
    pd.DataFrame(base).to_csv(ART / "baselines_decision.csv", index=False)

    return {"single_threshold": single, "band_unconstrained": band_free,
            "band_chosen": band, "costs_used": c,
            "test_prevalence": float(y.mean()), "n_test_images": int(len(y))}


def main() -> None:
    t0 = time.time()
    ART.mkdir(exist_ok=True)
    device = pick_device()
    print(f"[setup] device={device} epochs={EPOCHS} imgsz={IMGSZ} batch={BATCH} fraction={FRACTION}")

    data_yaml = download()
    names = class_names(data_yaml)
    c = decision.load_costs()
    missing = [p for p in c["positive_classes"] if p not in names]
    if missing:
        raise SystemExit(f"positive_classes {missing} not in dataset classes {names}")
    pos_idx = [names.index(p) for p in c["positive_classes"]]

    stats = describe(names)
    (ART / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print("[data]", json.dumps({k: v["instances"] for k, v in stats["splits"].items()}))

    weights = train(data_yaml, device)
    ev = evaluate(weights, data_yaml, names, device)
    ev["baselines_detection"] = [
        {"model": "Finetuned yolo11n", **ev["overall"]},
        {"model": "COCO yolo11n zero-shot", "precision": 0.0, "recall": 0.0, "AP50": 0.0,
         "mAP50-95": 0.0, "note": "COCO has no insulator/fault classes, so it cannot produce a single correct-class box"},
        {"model": "Trivial: predict nothing", "precision": 0.0, "recall": 0.0, "AP50": 0.0,
         "mAP50-95": 0.0, "note": "no boxes -> recall 0, AP 0"},
    ]
    (ART / "metrics.json").write_text(json.dumps(ev, indent=2))
    print("[test]", json.dumps(ev["per_class"], indent=1))

    df = score_test(weights, names, pos_idx, device)
    df.to_csv(ART / "test_scores.csv", index=False)

    op = choose_operating_point(df, c)
    op["train_settings"] = {"model": MODEL, "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH,
                            "fraction": FRACTION, "seed": SEED, "device": device,
                            "machine": platform.platform(), "cpu_count": os.cpu_count(),
                            "torch": torch.__version__, "minutes": round((time.time() - t0) / 60, 1)}
    (ART / "operating_point.json").write_text(json.dumps(op, indent=2, default=float))
    print("[decision]", json.dumps({k: op[k] for k in ("single_threshold", "band_chosen")},
                                   indent=1, default=float))
    print(f"[done] {op['train_settings']['minutes']} min. Commit weights/ and artifacts/.")


if __name__ == "__main__":
    main()
