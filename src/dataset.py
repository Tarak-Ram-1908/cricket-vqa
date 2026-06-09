from torch.utils.data import Dataset
from PIL import Image
import json

class CricketVQADataset(Dataset):
    def __init__(self, ann_path, img_root, processor):
        with open(ann_path) as f:
            coco = json.load(f)
        self.img_root = img_root
        self.processor = processor
        img_map = {i["id"]: i for i in coco["images"]}
        self.samples = [
            (img_map[a["image_id"]]["file_name"], a["question"], a["answer"])
            for a in coco["annotations"]
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, question, answer = self.samples[idx]
        image = Image.open(f"{self.img_root}/{fname}").convert("RGB")
        prompt = (
            f"USER: <image>\nYou are an expert cricket analyst. "
            f"Question: {question}\nASSISTANT:"
        )
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}, answer
