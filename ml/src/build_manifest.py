import csv
from pathlib import Path

OUTPUT_PATH = Path("data/manifest.csv")

BASE_DIR = "data/videos/Neurimo Synthetic Training Data"
NUM_CHILDREN = 167
AGE_BUCKET = "12_24"

rows = []

for i in range(NUM_CHILDREN):
    child_dir = f"{BASE_DIR}/Child{i:03d}"
    rows.append({
        "child_id": i,
        "age_bucket": AGE_BUCKET,
        "joint_attention_path": f"{child_dir}/{i}-JointAttention.mp4",
        "imitation_path": f"{child_dir}/{i}-Imitation.mp4",
        "free_play_path": f"{child_dir}/{i}-FreePlay.mp4",
    })

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "child_id",
            "age_bucket",
            "joint_attention_path",
            "imitation_path",
            "free_play_path",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Wrote manifest with {NUM_CHILDREN} children to {OUTPUT_PATH}")

