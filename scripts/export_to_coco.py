import json
from pathlib import Path
from PIL import Image

def ls_bbox_to_coco(ls_result, img_w, img_h):
    x = ls_result["x"] / 100
    y = ls_result["y"] / 100
    w = ls_result["width"] / 100
    h = ls_result["height"] / 100
    return [x + w/2, y + h/2, w, h]

def export(ls_export_path, frames_dir, out_path):
    with open(ls_export_path) as f:
        tasks = json.load(f)
    images, annotations = [], []
    ann_id = 1
    for task in tasks:
        img_path = task["data"]["image"].split("?d=")[-1]
        try:
            img = Image.open(img_path)
            img_w, img_h = img.size
        except:
            img_w, img_h = 1280, 720
        img_id = len(images) + 1
        images.append({
            "id": img_id,
            "file_name": str(Path(img_path).name),
            "width": img_w, "height": img_h,
            "match_meta": task.get("meta", {})
        })
        result = task["annotations"][0]["result"] if task.get("annotations") else []
        bboxes = {r["value"]["rectanglelabels"][0]: ls_bbox_to_coco(r["value"], img_w, img_h)
                  for r in result if r["type"] == "rectanglelabels"}
        texts = {r["from_name"]: r["value"]["text"][0]
                 for r in result if r["type"] == "textarea" and r["value"].get("text")}
        type_map = {
            "counting":  ("q_counting",  "a_counting"),
            "spatial":   ("q_spatial",   "a_spatial"),
            "formation": ("q_formation", "a_formation"),
            "context":   ("q_context",   "a_context"),
            "grounding": ("q_grounding", "a_grounding"),
        }
        for q_type, (qk, ak) in type_map.items():
            q = texts.get(qk, "")
            a = texts.get(ak, "")
            if not q or not a:
                continue
            bbox = bboxes.get("slip_cordon") if q_type == "grounding" else None
            annotations.append({
                "id": ann_id, "image_id": img_id,
                "question_id": ann_id, "question_type": q_type,
                "question": q, "answer": a, "bbox": bbox
            })
            ann_id += 1
    coco = {
        "info": {"dataset": "CricketVQA-v1", "version": "1.0", "year": 2026,
                 "contributor": "Alahari Tarak Ram"},
        "images": images, "annotations": annotations,
        "question_types": list(type_map.keys())
    }
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"Exported {len(images)} images, {len(annotations)} QA pairs to {out_path}")

if __name__ == "__main__":
    import sys
    export(sys.argv[1], sys.argv[2], sys.argv[3])
