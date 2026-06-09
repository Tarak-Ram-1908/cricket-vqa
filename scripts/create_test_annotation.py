import json, os

# Create a tiny 2-sample COCO JSON for testing
test_data = {
    "info": {"dataset": "CricketVQA-v1", "version": "1.0"},
    "images": [
        {"id": 1, "file_name": "test.jpg", "width": 1280, "height": 720,
         "match_meta": {"phase": "powerplay"}}
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "question_id": 1,
         "question_type": "counting",
         "question": "How many fielders are in the infield?",
         "answer": "4", "bbox": None}
    ],
    "question_types": ["counting","spatial","formation","context","grounding"]
}

os.makedirs("data/annotations", exist_ok=True)
with open("data/annotations/test_sample.json", "w") as f:
    json.dump(test_data, f, indent=2)
print("Test annotation file created.")
print(f"Sample: {test_data['annotations'][0]['question']}")
print(f"Answer: {test_data['annotations'][0]['answer']}")
