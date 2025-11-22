import os
import yaml
import glob
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO

# ============================================================
# 1. Compute Custom Anchors
# ============================================================
def compute_anchors(label_path, k=9):
    boxes = []

    for lbl in glob.glob(f"{label_path}/*.txt"):
        with open(lbl) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                _, _, _, w, h = map(float, parts)
                boxes.append([w, h])

    boxes = np.array(boxes)

    if len(boxes) == 0:
        print("No boxes found for anchor computation")
        return None

    kmeans = KMeans(n_clusters=k, random_state=0).fit(np.log(boxes + 1e-6))
    anchors = np.exp(kmeans.cluster_centers_).tolist()

    print("\n=== Custom Anchors Computed ===")
    for a in anchors:
        print(a)

    return anchors

# ============================================================
# 2. Update data.yaml with anchors
# ============================================================
def update_yaml_with_anchors(anchors, yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # YOLO requires anchors in 3 groups
    data["anchors"] = [
        anchors[0:3],
        anchors[3:6],
        anchors[6:9]
    ]

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)

    print("\n=== data.yaml updated with anchors ===")

# ============================================================
# 3. Freeze model backbone
# ============================================================
def freeze_backbone(model):
    for m in model.model.modules():
        m.requires_grad = False
    for m in model.model.head.modules():
        m.requires_grad = True

# ============================================================
# 4. Unfreeze full model
# ============================================================
def unfreeze_all(model):
    for m in model.model.modules():
        m.requires_grad = True

# ============================================================
# 5. MAIN TRAINING PIPELINE
# ============================================================
def main():

    DATA_YAML = "data.yaml"   # your YAML file
    TRAIN_LABELS = "train_2/train2/images"  # update if needed
    DEVICE = 0  # 0 for GPU, "cpu" for CPU

    print("\n====================================")
    print("STAGE 0 — COMPUTING ANCHORS")
    print("====================================")

    anchors = compute_anchors(TRAIN_LABELS, k=9)
    if anchors:
        update_yaml_with_anchors(anchors, DATA_YAML)

    # ============================================================
    # STAGE 1 — FROZEN TRAINING (warmup)
    # ============================================================
    print("\n====================================")
    print("STAGE 1 — FROZEN BACKBONE TRAINING")
    print("====================================")

    model = YOLO("yolov8s.pt")
    freeze_backbone(model)

    model.train(
        data=DATA_YAML,
        epochs=20,
        batch=16,
        imgsz=640,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.1,
        cos_lr=True,
        patience=20,
        amp=True,
        mosaic=1.0,
        mixup=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        degrees=5,
        translate=0.1,
        scale=0.5,
        shear=1.0,
        close_mosaic=10,
        device=DEVICE
    )

    # ============================================================
    # STAGE 2 — FULL TRAINING (unfrozen + SAFE EARLY STOP)
    # ============================================================
    print("\n====================================")
    print("STAGE 2 — FULL TRAINING (UNFROZEN + EARLY STOP)")
    print("====================================")

    unfreeze_all(model)

    model.train(
        data=DATA_YAML,
        epochs=150,        # max limit
        batch=16,
        imgsz=640,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.05,
        cos_lr=True,
        patience=30,       # SAFE early stopping
        amp=True,
        ema=True,
        mosaic=0.8,
        mixup=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        degrees=5,
        translate=0.1,
        scale=0.6,
        shear=1.5,
        perspective=0.0005,
        close_mosaic=15,
        device=DEVICE
    )

    print("\n====================================")
    print("TRAINING COMPLETE — BEST ACCURACY + SAFETY EARLY STOP")
    print("====================================")

    # ============================================================
    # EVALUATION ON TEST SET
    # ============================================================
    print("\n=== EVALUATING ON TEST SET ===")
    best_model_path = "runs/train/exp/weights/best.pt"  # default YOLO save path
    if not os.path.exists(best_model_path):
        # fallback: check for latest model folder
        best_model_path = max(glob.glob("runs/train/*/weights/best.pt"), key=os.path.getctime)

    model = YOLO(best_model_path)
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        tta=True
    )

    print("\n=== TEST METRICS ===")
    print(metrics)
    print("===============================")


if __name__ == "__main__":
    main()
