"""
app/ml/inference.py

✅ ของเดิม: predict() — return Top 1 (ไม่แตะ ใช้ได้ปกติ)
🆕 เพิ่มใหม่: predict_top3() — return Top 3 เรียงตามค่าความมั่นใจ

วิธีใช้: แทนที่ไฟล์ inference.py เดิม
"""
import torch
from torchvision import transforms
import json
import os

BASE_DIR = os.path.dirname(__file__)

with open(os.path.join(BASE_DIR, "classes.json"), "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# =============================================================
# ✅ ฟังก์ชันเดิม — ไม่แก้ไข (ใช้กับ predict_router เดิมได้ปกติ)
# =============================================================
def predict(model, image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return {
        "class_name": CLASS_NAMES[predicted.item()],
        "confidence": round(confidence.item(), 4)
    }


# =============================================================
# 🆕 ฟังก์ชันใหม่ — return Top 3 เรียงจากมากไปน้อย
# =============================================================
def predict_top3(model, image, top_k=3):
    """
    จำแนกภาพแล้วคืนผล Top K (default = 3)

    Returns:
        list of dict, เรียงตาม confidence จากมากไปน้อย
        [
            {"class_name": "Para cress", "confidence": 0.9812, "rank": 1},
            {"class_name": "Neem tree", "confidence": 0.0134, "rank": 2},
            {"class_name": "RattailedRadish", "confidence": 0.0029, "rank": 3},
        ]
    """
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        # ดึง Top K
        top_confidences, top_indices = torch.topk(probabilities, k=min(top_k, len(CLASS_NAMES)), dim=1)

    results = []
    for rank, (conf, idx) in enumerate(
        zip(top_confidences[0], top_indices[0]), start=1
    ):
        results.append({
            "class_name": CLASS_NAMES[idx.item()],
            "confidence": round(conf.item(), 4),
            "rank": rank,
        })

    return results
