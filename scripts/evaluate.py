"""
evaluate.py — Cricket VQA Evaluation Suite
==========================================
Metrics:
  - VQA Accuracy      : exact-match + soft-match per question type
  - CIDEr             : for open-ended types (formation)
  - BLEU-1/4          : for formation type
  - Consistency Score : same frame, paraphrased question → same answer?
  - Counting MAE/Acc  : exact integer match + ±1 tolerance
  - Spatial F1        : precision/recall for yes/no + position name answers
  - Grounding IoU     : bbox overlap for grounding type

Usage:
  python scripts/evaluate.py \
      --predictions data/eval/predictions.json \
      --ground_truth data/annotations/cricketvqa_v1.json \
      --output_dir  data/eval/results

Input format — predictions.json:
  [
    {
      "question_id": 1,
      "question_type": "counting",   # counting | spatial | formation | context | grounding
      "predicted_answer": "4",
      "image_id": 1
    },
    ...
  ]

Ground truth — cricketvqa_v1.json (standard CricketVQA COCO format):
  {
    "annotations": [
      {
        "id": 1,
        "image_id": 1,
        "question_id": 1,
        "question_type": "counting",
        "question": "...",
        "answer": "4",
        "bbox": null          # [x_c, y_c, w, h] normalised for grounding
      },
      ...
    ]
  }
"""

import json
import math
import re
import argparse
import os
from collections import defaultdict
from typing import Any

# ── optional heavy deps (graceful fallback) ──────────────────────────────────
try:
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    HAS_COCO_EVAL = True
except ImportError:
    HAS_COCO_EVAL = False
    print("[WARN] pycocoevalcap not found — CIDEr/BLEU will be skipped.")
    print("       Install with: pip install pycocoevalcap")

try:
    from sklearn.metrics import f1_score, precision_score, recall_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] scikit-learn not found — F1 metrics will be computed manually.")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

VALID_POSITIONS = {
    "slip", "gully", "point", "cover-point", "cover", "extra-cover",
    "mid-off", "mid-on", "mid-wicket", "square-leg", "fine-leg",
    "third-man", "deep-point", "sweeper-cover", "long-on", "long-off",
    "deep-fine-leg", "short-leg", "cannot-determine",
    # common aliases
    "deep square leg", "square leg", "backward point", "deep cover",
    "deep mid-wicket", "cow corner",
}

CONTEXT_LABELS = {"powerplay", "middle", "death", "attacking", "defensive"}


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def parse_int(text: str) -> int | None:
    """Extract first integer from a string."""
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def parse_bbox(text: str) -> list[float] | None:
    """Parse '[x_c, y_c, w, h]' string → list of 4 floats."""
    nums = re.findall(r"[\d.]+", text)
    if len(nums) == 4:
        return [float(n) for n in nums]
    return None


def iou(box_a: list[float], box_b: list[float]) -> float:
    """
    Compute IoU for two boxes in [x_center, y_center, width, height] format,
    all values normalised 0–1.
    """
    def to_corners(b):
        x1 = b[0] - b[2] / 2
        y1 = b[1] - b[3] / 2
        x2 = b[0] + b[2] / 2
        y2 = b[1] + b[3] / 2
        return x1, y1, x2, y2

    ax1, ay1, ax2, ay2 = to_corners(box_a)
    bx1, by1, bx2, by2 = to_corners(box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area_a  = (ax2 - ax1) * (ay2 - ay1)
    area_b  = (bx2 - bx1) * (by2 - by1)
    union   = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PER-TYPE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 2a. Counting ─────────────────────────────────────────────────────────────

def eval_counting(preds: list[dict], gts: dict[int, dict]) -> dict:
    """
    Metrics:
      exact_acc  — predicted int == ground-truth int
      tolerance1 — |pred - gt| <= 1
      mae        — mean absolute error
      samples    — number evaluated
    """
    exact, tol1, abs_errors, skipped = 0, 0, [], 0

    for p in preds:
        qid  = p["question_id"]
        gt   = gts.get(qid)
        if gt is None:
            skipped += 1
            continue

        pred_int = parse_int(str(p["predicted_answer"]))
        gt_int   = parse_int(str(gt["answer"]))

        if pred_int is None or gt_int is None:
            skipped += 1
            continue

        diff = abs(pred_int - gt_int)
        abs_errors.append(diff)
        if diff == 0:
            exact += 1
        if diff <= 1:
            tol1 += 1

    n = len(abs_errors)
    return {
        "exact_acc":  round(exact / n, 4) if n else 0,
        "tolerance1": round(tol1  / n, 4) if n else 0,
        "mae":        round(sum(abs_errors) / n, 4) if n else 0,
        "samples":    n,
        "skipped":    skipped,
    }


# ── 2b. Spatial ──────────────────────────────────────────────────────────────

def eval_spatial(preds: list[dict], gts: dict[int, dict]) -> dict:
    """
    Metrics:
      exact_acc   — full exact match after normalisation
      yes_no_acc  — accuracy only on yes/no answers
      position_acc— accuracy only on position-name answers
      f1          — macro F1 across all closed-set labels
    """
    exact = 0
    yn_correct, yn_total = 0, 0
    pos_correct, pos_total = 0, 0
    all_labels, all_preds_f1 = [], []
    skipped = 0

    for p in preds:
        qid = p["question_id"]
        gt  = gts.get(qid)
        if gt is None:
            skipped += 1
            continue

        pred = normalise(str(p["predicted_answer"]))
        gt_a = normalise(str(gt["answer"]))

        if pred == gt_a:
            exact += 1

        all_labels.append(gt_a)
        all_preds_f1.append(pred)

        if gt_a in ("yes", "no"):
            yn_total += 1
            if pred == gt_a:
                yn_correct += 1
        elif gt_a in VALID_POSITIONS:
            pos_total += 1
            if pred == gt_a:
                pos_correct += 1

    n = len(all_labels)

    # Manual macro F1 (no sklearn dependency needed for binary yes/no subset)
    yn_f1 = _binary_f1(all_labels, all_preds_f1)

    return {
        "exact_acc":    round(exact      / n,          4) if n          else 0,
        "yes_no_acc":   round(yn_correct / yn_total,   4) if yn_total   else 0,
        "position_acc": round(pos_correct/ pos_total,  4) if pos_total  else 0,
        "yes_no_f1":    yn_f1,
        "samples":      n,
        "skipped":      skipped,
    }


def _binary_f1(labels: list[str], preds: list[str]) -> float:
    """Compute F1 only for yes/no pairs; ignore others."""
    yn_pairs = [(l, p) for l, p in zip(labels, preds) if l in ("yes", "no")]
    if not yn_pairs:
        return 0.0
    ls, ps = zip(*yn_pairs)
    # Map yes→1, no→0, other→-1
    def to_int(v):
        return 1 if v == "yes" else (0 if v == "no" else -1)
    ls_i = [to_int(x) for x in ls]
    ps_i = [to_int(x) for x in ps]

    if HAS_SKLEARN:
        return round(float(f1_score(ls_i, ps_i, average="binary", pos_label=1,
                                    zero_division=0)), 4)
    # Manual fallback
    tp = sum(l == 1 and p == 1 for l, p in zip(ls_i, ps_i))
    fp = sum(l == 0 and p == 1 for l, p in zip(ls_i, ps_i))
    fn = sum(l == 1 and p == 0 for l, p in zip(ls_i, ps_i))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    return round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0


# ── 2c. Formation (open-ended) ───────────────────────────────────────────────

def eval_formation(preds: list[dict], gts: dict[int, dict]) -> dict:
    """
    Metrics:
      cider  — CIDEr score (requires pycocoevalcap)
      bleu1  — BLEU-1
      bleu4  — BLEU-4
      exact  — exact match (rare but possible)
    """
    refs, hyps = {}, {}
    exact = 0
    skipped = 0

    for p in preds:
        qid = p["question_id"]
        gt  = gts.get(qid)
        if gt is None:
            skipped += 1
            continue

        pred = normalise(str(p["predicted_answer"]))
        gt_a = normalise(str(gt["answer"]))

        if pred == gt_a:
            exact += 1

        refs[str(qid)] = [gt_a]
        hyps[str(qid)] = [pred]

    n = len(refs)
    result: dict[str, Any] = {"exact_acc": round(exact / n, 4) if n else 0,
                               "samples": n, "skipped": skipped}

    if HAS_COCO_EVAL and n > 0:
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(refs, hyps)
        result["cider"] = round(float(cider_score), 4)

        bleu_scorer = Bleu(4)
        bleu_scores, _ = bleu_scorer.compute_score(refs, hyps)
        result["bleu1"] = round(float(bleu_scores[0]), 4)
        result["bleu4"] = round(float(bleu_scores[3]), 4)
    else:
        # Simple unigram overlap as CIDEr proxy
        result["unigram_overlap"] = _mean_unigram_overlap(refs, hyps)

    return result


def _mean_unigram_overlap(refs: dict, hyps: dict) -> float:
    """Macro-average unigram F1 across all pairs (CIDEr proxy if pycocoeval unavailable)."""
    scores = []
    for k in refs:
        ref_toks  = set(refs[k][0].split())
        hyp_toks  = set(hyps[k][0].split())
        if not ref_toks:
            continue
        overlap = len(ref_toks & hyp_toks)
        prec = overlap / len(hyp_toks) if hyp_toks else 0
        rec  = overlap / len(ref_toks)
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        scores.append(f1)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


# ── 2d. Context ──────────────────────────────────────────────────────────────

def eval_context(preds: list[dict], gts: dict[int, dict]) -> dict:
    """
    3-class accuracy: powerplay / middle / death (and attacking/defensive).
    Also reports per-class accuracy.
    """
    correct = 0
    per_class: dict[str, list] = defaultdict(list)   # label → [correct?]
    skipped = 0

    for p in preds:
        qid = p["question_id"]
        gt  = gts.get(qid)
        if gt is None:
            skipped += 1
            continue

        pred = normalise(str(p["predicted_answer"]))
        gt_a = normalise(str(gt["answer"]))

        hit = int(pred == gt_a)
        correct += hit
        per_class[gt_a].append(hit)

    n = len([q for q in gts.values()]) - skipped
    total = sum(len(v) for v in per_class.values())

    return {
        "overall_acc": round(correct / total, 4) if total else 0,
        "per_class":   {k: round(sum(v) / len(v), 4) for k, v in per_class.items()},
        "samples":     total,
        "skipped":     skipped,
    }


# ── 2e. Grounding ────────────────────────────────────────────────────────────

def eval_grounding(preds: list[dict], gts: dict[int, dict]) -> dict:
    """
    Metrics:
      mean_iou   — average IoU across all grounding pairs
      iou_50     — fraction with IoU >= 0.50
      iou_25     — fraction with IoU >= 0.25
    """
    ious = []
    skipped = 0

    for p in preds:
        qid = p["question_id"]
        gt  = gts.get(qid)
        if gt is None or gt.get("bbox") is None:
            skipped += 1
            continue

        pred_bbox = parse_bbox(str(p["predicted_answer"]))
        gt_bbox   = gt["bbox"]

        if pred_bbox is None or len(gt_bbox) != 4:
            skipped += 1
            continue

        ious.append(iou(pred_bbox, gt_bbox))

    n = len(ious)
    return {
        "mean_iou": round(sum(ious) / n, 4)                        if n else 0,
        "iou_50":   round(sum(v >= 0.50 for v in ious) / n, 4)    if n else 0,
        "iou_25":   round(sum(v >= 0.25 for v in ious) / n, 4)    if n else 0,
        "samples":  n,
        "skipped":  skipped,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONSISTENCY SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def eval_consistency(preds: list[dict], gts: dict[int, dict],
                     consistency_pairs: list[tuple[int, int]] | None = None) -> dict:
    """
    Consistency: given two different questions about the same image that should
    have the same answer (e.g. "how many fielders in infield?" vs "count infield
    fielders"), do the model predictions agree?

    consistency_pairs: list of (question_id_A, question_id_B) tuples.
    If not provided, this metric returns None and is skipped.

    Consistency score = fraction of pairs where predicted answers match.
    """
    if not consistency_pairs:
        return {"consistency": None,
                "note": "No consistency pairs provided. "
                        "Pass --consistency_pairs path/to/pairs.json"}

    agree = 0
    evaluated = 0
    pred_lookup = {p["question_id"]: normalise(str(p["predicted_answer"]))
                   for p in preds}

    for qid_a, qid_b in consistency_pairs:
        pred_a = pred_lookup.get(qid_a)
        pred_b = pred_lookup.get(qid_b)
        if pred_a is None or pred_b is None:
            continue
        evaluated += 1
        if pred_a == pred_b:
            agree += 1

    return {
        "consistency": round(agree / evaluated, 4) if evaluated else 0,
        "pairs_evaluated": evaluated,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. OVERALL VQA ACCURACY  (weighted average across types)
# ═══════════════════════════════════════════════════════════════════════════════

def overall_vqa_accuracy(type_results: dict[str, dict]) -> float:
    """
    Weighted average accuracy:
      counting  → exact_acc
      spatial   → exact_acc
      formation → exact_acc (or unigram_overlap)
      context   → overall_acc
      grounding → iou_50
    All weights = 1 (equal contribution per type).
    """
    score_map = {
        "counting":  type_results.get("counting",  {}).get("exact_acc",      0),
        "spatial":   type_results.get("spatial",   {}).get("exact_acc",      0),
        "formation": type_results.get("formation", {}).get("exact_acc",      0),
        "context":   type_results.get("context",   {}).get("overall_acc",    0),
        "grounding": type_results.get("grounding", {}).get("iou_50",         0),
    }
    values = [v for v in score_map.values() if v is not None]
    return round(sum(values) / len(values), 4) if values else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def load_ground_truth(gt_path: str) -> dict[int, dict]:
    """Load annotations from cricketvqa_v1.json, keyed by question_id."""
    with open(gt_path) as f:
        data = json.load(f)
    return {a["question_id"]: a for a in data["annotations"]}


def load_predictions(pred_path: str) -> dict[str, list[dict]]:
    """Load predictions, return dict keyed by question_type."""
    with open(pred_path) as f:
        preds = json.load(f)
    by_type: dict[str, list] = defaultdict(list)
    for p in preds:
        by_type[p["question_type"]].append(p)
    return by_type


def run_evaluation(predictions_path: str,
                   ground_truth_path: str,
                   output_dir: str,
                   consistency_pairs_path: str | None = None) -> dict:

    os.makedirs(output_dir, exist_ok=True)

    gt_all    = load_ground_truth(ground_truth_path)
    preds_by_type = load_predictions(predictions_path)

    # Split GT by type for convenience
    gt_by_type: dict[str, dict] = defaultdict(dict)
    for qid, ann in gt_all.items():
        gt_by_type[ann["question_type"]][qid] = ann

    # ── run per-type evaluators ──────────────────────────────────────────────
    results: dict[str, Any] = {}

    results["counting"]  = eval_counting(
        preds_by_type.get("counting", []), gt_by_type["counting"])

    results["spatial"]   = eval_spatial(
        preds_by_type.get("spatial", []),  gt_by_type["spatial"])

    results["formation"] = eval_formation(
        preds_by_type.get("formation", []), gt_by_type["formation"])

    results["context"]   = eval_context(
        preds_by_type.get("context", []),  gt_by_type["context"])

    results["grounding"] = eval_grounding(
        preds_by_type.get("grounding", []), gt_by_type["grounding"])

    # ── consistency ─────────────────────────────────────────────────────────
    pairs = None
    if consistency_pairs_path and os.path.exists(consistency_pairs_path):
        with open(consistency_pairs_path) as f:
            pairs = [tuple(p) for p in json.load(f)]

    all_preds_flat = [p for ps in preds_by_type.values() for p in ps]
    results["consistency"] = eval_consistency(all_preds_flat, gt_all, pairs)

    # ── overall VQA accuracy ────────────────────────────────────────────────
    results["overall_vqa_accuracy"] = overall_vqa_accuracy(results)

    # ── save results ─────────────────────────────────────────────────────────
    out_path = os.path.join(output_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    _print_summary(results)
    print(f"\nFull results saved → {out_path}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PRETTY PRINTER
# ═══════════════════════════════════════════════════════════════════════════════

def _print_summary(results: dict):
    LINE = "─" * 56
    print(f"\n{'═'*56}")
    print(f"  CRICKET VQA — EVALUATION RESULTS")
    print(f"{'═'*56}")

    def row(label, value, indent=2):
        pad = " " * indent
        val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"{pad}{label:<34} {val_str}")

    # counting
    print(f"\n{LINE}")
    print("  TYPE 1 — COUNTING")
    c = results.get("counting", {})
    row("Exact Accuracy",       c.get("exact_acc", "—"))
    row("±1 Tolerance Accuracy",c.get("tolerance1", "—"))
    row("Mean Absolute Error",  c.get("mae", "—"))
    row("Samples",              c.get("samples", 0))

    # spatial
    print(f"\n{LINE}")
    print("  TYPE 2 — SPATIAL")
    s = results.get("spatial", {})
    row("Exact Accuracy",   s.get("exact_acc", "—"))
    row("Yes/No Accuracy",  s.get("yes_no_acc", "—"))
    row("Position Accuracy",s.get("position_acc", "—"))
    row("Yes/No F1",        s.get("yes_no_f1", "—"))
    row("Samples",          s.get("samples", 0))

    # formation
    print(f"\n{LINE}")
    print("  TYPE 3 — FORMATION (open-ended)")
    fm = results.get("formation", {})
    row("Exact Accuracy",   fm.get("exact_acc", "—"))
    if "cider" in fm:
        row("CIDEr",        fm.get("cider", "—"))
        row("BLEU-1",       fm.get("bleu1", "—"))
        row("BLEU-4",       fm.get("bleu4", "—"))
    else:
        row("Unigram Overlap (CIDEr proxy)", fm.get("unigram_overlap", "—"))
    row("Samples",          fm.get("samples", 0))

    # context
    print(f"\n{LINE}")
    print("  TYPE 4 — CONTEXT")
    ctx = results.get("context", {})
    row("Overall Accuracy", ctx.get("overall_acc", "—"))
    for lbl, acc in ctx.get("per_class", {}).items():
        row(f"  {lbl}", acc, indent=4)
    row("Samples",          ctx.get("samples", 0))

    # grounding
    print(f"\n{LINE}")
    print("  TYPE 5 — GROUNDING (bbox IoU)")
    g = results.get("grounding", {})
    row("Mean IoU",  g.get("mean_iou", "—"))
    row("IoU ≥ 0.50",g.get("iou_50",  "—"))
    row("IoU ≥ 0.25",g.get("iou_25",  "—"))
    row("Samples",   g.get("samples", 0))

    # consistency
    print(f"\n{LINE}")
    print("  CONSISTENCY")
    con = results.get("consistency", {})
    if con.get("consistency") is not None:
        row("Consistency Score", con.get("consistency", "—"))
        row("Pairs Evaluated",   con.get("pairs_evaluated", 0))
    else:
        print(f"  {con.get('note', 'No pairs provided')}")

    # overall
    print(f"\n{'═'*56}")
    ova = results.get("overall_vqa_accuracy", 0)
    print(f"  OVERALL VQA ACCURACY (equal-weight avg)   {ova:.4f}")
    print(f"{'═'*56}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cricket VQA Evaluation — CIDEr, VQA Acc, Consistency")
    parser.add_argument("--predictions",        required=True,
                        help="Path to model predictions JSON")
    parser.add_argument("--ground_truth",       required=True,
                        help="Path to cricketvqa_v1.json ground truth")
    parser.add_argument("--output_dir",         default="data/eval/results",
                        help="Directory to save eval_results.json")
    parser.add_argument("--consistency_pairs",  default=None,
                        help="Optional JSON list of [qid_A, qid_B] pairs for "
                             "consistency scoring")
    args = parser.parse_args()

    run_evaluation(
        predictions_path     = args.predictions,
        ground_truth_path    = args.ground_truth,
        output_dir           = args.output_dir,
        consistency_pairs_path = args.consistency_pairs,
    )
