"""
generate_consistency_pairs.py
==============================
Automatically generates consistency pairs from your annotation JSON.

A "consistency pair" = two different question_ids that ask about the same
thing on the same image so the model should give the same answer.

Strategy used here:
  - Find all QA pairs of the same question_type on the same image
  - Pair up any two questions whose ground-truth answers are identical
  - Save as [[qid_A, qid_B], ...] JSON

Usage:
  python scripts/generate_consistency_pairs.py \
      --annotations data/annotations/cricketvqa_v1.json \
      --output      data/eval/consistency_pairs.json

Then pass to evaluate.py:
  python scripts/evaluate.py ... \
      --consistency_pairs data/eval/consistency_pairs.json
"""

import json
import argparse
from collections import defaultdict
from itertools import combinations


def generate_pairs(annotations_path: str, output_path: str):
    with open(annotations_path) as f:
        data = json.load(f)

    anns = data["annotations"]

    # Group by (image_id, question_type, normalised_answer)
    bucket: dict[tuple, list[int]] = defaultdict(list)
    for a in anns:
        key = (a["image_id"], a["question_type"],
               a["answer"].lower().strip())
        bucket[key].append(a["question_id"])

    pairs = []
    for key, qids in bucket.items():
        if len(qids) >= 2:
            # All pairwise combinations within this bucket
            for qid_a, qid_b in combinations(qids, 2):
                pairs.append([qid_a, qid_b])

    with open(output_path, "w") as f:
        json.dump(pairs, f)

    print(f"Generated {len(pairs)} consistency pairs → {output_path}")
    if not pairs:
        print("  No pairs found. This means each image has at most one "
              "question per type — that's fine at this dataset size.\n"
              "  To add consistency pairs manually: when annotating, "
              "annotate the same frame twice with paraphrased questions "
              "and identical expected answers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", default="data/eval/consistency_pairs.json")
    args = parser.parse_args()
    generate_pairs(args.annotations, args.output)
