# Assignment 2 Report

- **Name**: Roberto Falcato
- **Student ID**: 20219
- **Email**: rfalcato.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong

### The decision
A regional grid operator has a network of approximately 10,000 km of high voltage overhead lines. An annual inspection is performed using drones, this equates to approximately 45km of lines to inspect each day across the 220 working days per year. As there are approximately 3 towers per km and 6 insulators per tower this amounts to approximately 800 images of insulators. These are currently reviewed by an engineer. The model takes over this task and classifies the images into 3 categories:

| outcome | action |
|---|---|
| **Pass** | insulator left in service |
| **Engineer review** | a grid engineer checks the photo at a desk (~45 s) |
| **Dispatch** | a line crew drives to the tower and replaces the insulator |

### Who acts on it
A grid engineer works through the review queue, a crew of ~3 linemen with a truck carries out
dispatches, maintenance planning owns the thresholds.

### False negative: a damaged insulator is passed (€20,000)
The insulator remains on the line until it is reviewed next year. The string may fail due to cracking or breaking which can cause a flashover. This causes a failure on the line. This can cause an outage, damage to equipment and in the worst case, a fire. I have calculated this as an expected value:

- P(Visually damaged string failing in 12 months) = 20% (Assumption, could not find public source)
- Cost of failure:
  - Emergency repair: ~€15,000
  - Expected loss of supply: ~€60,000 (20% chance of interruption x 10MW x 2 hrs x Value of lost load (~€15,000/MWh) between Belgium and Ireland regulatory estimates)
  - Safety, Fire and Regulatory risk: ~€25,000 (Assumption)
- Expected Cost: 0.20 x €100,000 = €20,000 per fault missed

### False positive: a healthy insulator is dispatched (€1,500)
1 day of work is wasted by the line crew. 3 linemen x 8hrs x ~€60/hr + Truck and travel = ~€1,500. This ties in with Powerlink's (Queensland) live line insulator replacement programme. This is approximately €1,400~€1,800 per string. If the dispatch was a true positive then the cost would be the same. The new insulator would have been purchased in both cases, therefore this is not included.

### Why the asymmetry matters
A miss costs about **13×** a false alarm. A crew is therefore worth

## Dataset

- **Roboflow Universe:** workspace `insulators-ljwyo`, project `insulators-c4gwk`, **version 1**
- **Link:** https://universe.roboflow.com/insulators-ljwyo/insulators-c4gwk/dataset/1
- **License:** CC BY 4.0. Downloaded as-is, not forked.
- **Classes:** insulator, defect, brokeninsulator, brokenpart. Positive = any of defect / brokeninsulator / brokenpart.

**What one image is:** 
One aerial photo of a single insulator string on a transmission tower, with bounding boxes for:

| class | what the box marks |
|---|---|
| `insulator` | the whole insulator string |
| `brokeninsulator` | a string that is visibly damaged |
| `brokenpart` | the missing or shattered disc(s) on that string |
| `defect` | surface damage such as burn or flashover marks, without missing pieces |

For the decision, a photo is **positive (faulty)** if it contains at least one `defect`,
`brokeninsulator` or `brokenpart` box. Its score is the model's highest confidence on any of those
three classes.

**Why it serves the decision (and where it only resembles it):** 
It serves the decision because the image is the same kind of evidence the inspection team already
uses: a drone photo of one insulator string, with damage marked at the level a crew would act on.

It only *resembles* production in three ways:
1. **The photos are close-ups.** The median box covers 26–49% of the image, so the insulator fills
   the frame. A real inspection photo of a whole tower shows each insulator much smaller.
2. **It was collected to show faults.** 86% of test photos contain a fault, against an assumed ~2%
   on a real line. That's why every cost is re-weighted to the 2% base rate rather than using raw
   test counts.
3. **Each photo appears ~2.8 times** as flipped copies (see below), so the dataset holds less
   independent information than its image count suggests.

## What the dataset's composition implies

The published version has 958 train and 141 valid images, and no test split. The filenames also
show that each photo was saved several times as flipped copies (`795`, `795h`, `795v`): the 1,099
images come from only **397 original photos**, and **101 of them had copies in both train and
valid**, which would inflate the scores. So I re-split 70/15/15 by original photo (seed 0), keeping
all copies together.

| split | images | insulator | defect | brokeninsulator | brokenpart |
|---|---|---|---|---|---|
| train | 771 | 560 | 112 | 501 | 646 |
| valid | 173 | 87 | 18 | 131 | 153 |
| test | 155 | 70 | **16** | 118 | 144 |
| median box (% of image) | | 39% | 26% | 49% | 28% |

What this means:
- **The real sample is ~400 photos**, not 1,099, so results are less certain than they look.
- **`defect` is too thin to trust**: 16 test boxes, so one or two photos move its recall a lot.
- **The test set is 86% faulty** (only ~22 healthy photos), while a real line is ~2% faulty. The
  false-alarm rate, which drives most of the cost, rests on very few photos.
- **The photos are close-ups**, so performance on real whole-tower photos is likely lower.

## Finetuning and baselines

I finetuned `yolo11n` (detection, COCO-pretrained). Detection shows the engineer *where* the damage
is; the nano model trains on a CPU in under an hour. Trained in GitHub Codespaces (4-core CPU):
20 epochs, 512 px, batch 16, seed 0, 51.6 min in total.

| model (same 155 test photos) | AP50 | mAP50-95 | AUC | best cost / 1,000 |
|---|---|---|---|---|
| **Finetuned yolo11n** | **0.752** | **0.630** | **0.864** | **€196,969** |
| COCO, no finetuning | 0 | 0 | 0.484 | €394,436 |
| Predict nothing | 0 | 0 | 0.500 | €400,000 |

COCO has never seen an insulator, so it can't draw a correct box, and on the decision it does no
better than chance. The finetuned model ranks a faulty photo above a healthy one 86% of the time and
roughly halves the cost of doing nothing. That gap is what I expected.

## Per-class evaluation and failure modes

Positive = a photo with a `defect`, `brokeninsulator` or `brokenpart`.

| class | precision | recall | AP50 | mAP50-95 |
|---|---|---|---|---|
| brokeninsulator | 0.914 | 0.873 | **0.961** | 0.884 |
| brokenpart | 0.885 | 0.685 | 0.771 | 0.631 |
| defect | 0.813 | **0.375** | **0.467** | 0.252 |
| insulator | 0.747 | 0.943 | 0.811 | 0.754 |
| **overall** | 0.840 | 0.719 | 0.752 | 0.630 |

The headline comes mostly from `brokeninsulator` and `brokenpart`. `defect` looks like the weak
spot (recall 0.375), but the confusion matrix shows that 9 of the 16 defects were labelled
`brokenpart` and only 2 were missed completely, so most defect photos are still flagged as faulty.
The real confusion is between broken and healthy strings: 47 broken strings were called a plain
`insulator` and 26 healthy ones were called broken. The model judges the whole string poorly and
catches damage mainly through the `brokenpart` box.

## Cost matrix and operating point

| outcome | € | basis |
|---|---|---|
| False negative | 20,000 | 20% chance of failing before next inspection (my assumption) × ~€100k per failure (emergency repair, lost power valued at ~€15,000/MWh, safety/fire risk) |
| False positive | 1,500 | one wasted crew day (3 linemen × 8 h × ~€60/h + truck), in line with Powerlink's ~€1,400–1,800 per string |
| True positive | 1,500 | same trip; the part is paid either way |
| True negative | 0 | |
| Review | 0.63 | 45 s of an engineer at €50/h |
| Base rate | 2% | assumed share of damaged insulators on a real line |

I tested how sensitive this is. The review band [0.04, 0.96) does not change at all for a
false-negative cost anywhere between €2,000 and €20,000, or a base rate between 0.5% and 5%,
because it is set by the one engineer's capacity rather than by the costs. Only the model-alone
threshold moves (from 0.83 to 0.96) if the false-negative cost drops below about €10,000 or the
base rate below 1%. So the choice is robust to my two least certain assumptions.

## What a person still has to check

Below **0.04** → pass. **0.04–0.96** → engineer review. **0.96 or above** → dispatch.

That sends 59% of photos (~473 a day) to review: about 5.9 hours, **exactly one engineer**, which
is all we have. It's realistic but leaves no slack.

| strategy | € / 1,000 | € / day | engineers |
|---|---|---|---|
| Do nothing | 400,000 | 320,000 | 0 |
| Model alone | 196,969 | 157,575 | 0 |
| **Model + review** | **58,189** | **46,552** | **0.99** |
| Review everything | 30,625 | 24,500 | 1.67 |

The review band cuts the model-alone cost by 70%. Reviewing everything would save another ~€22,000
a day but needs a second engineer, who costs ~€400 a day. So the operator should **hire one**, and
use the model to put the most suspicious photos first. This assumes reviewers don't make mistakes.

## Limitations & next steps

- **Close-up photos**: test on real whole-tower photos, or detect the insulator first and zoom in.
- **Few defects and few healthy test photos**: label more, and add confidence intervals.
- **Assumed costs, perfect reviewers, one cost for all faults**: use real failure data, a reviewer
  error rate, and a separate cost per fault type.
- **One run of the smallest model**: repeat with more seeds and try `yolo11s` on a GPU.

## Generative AI use disclosure

I used Claude to compare datasets, write the pipeline and dashboard code (which I ran and can explain), research cost benchmarks and after writing the report on my own I asked it to transcribe it into markdown format to paste it correctly into Github. The framing, costs, thresholds and conclusions are mine.
