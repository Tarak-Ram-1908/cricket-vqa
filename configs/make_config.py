import yaml

config = {
    "model": "llava-hf/llava-1.5-7b-hf",
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 0.0002,
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "epochs": 3,
    "warmup_steps": 100,
    "checkpoint_dir": "checkpoints/",
    "ann_path": "data/annotations/cricketvqa_v1.json",
    "img_root": "data/frames"
}

with open("configs/train_config.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)

print("Config created.")
