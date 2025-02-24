import cv2
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
import numpy as np


### Common Setup ###

# Path to your trained model directory
MODEL_DIR = "./my_trained_yolos_model_v2"

# Load the feature extractor and model once
print("Loading model and feature extractor...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)
model = AutoModelForObjectDetection.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Mapping from category IDs to names
ID_TO_NAME = {
    0: "eur-1-2-5-10-20-50-100-200",
    1: "1",
    2: "10",
    3: "100",
    4: "100eur",
    5: "10eur",
    6: "2",
    7: "20",
    8: "200",
    9: "200eur",
    10: "20eur",
    11: "5",
    12: "50",
    13: "50eur",
    14: "5eur"
}

def run_inference(image: np.ndarray):
    """
    Run inference on an image (as a NumPy array), annotate detections,
    and return the annotated image along with counts for bills and coins.
    """
    # Convert BGR to RGB and prepare for inference
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    inputs = feature_extractor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # YOLOS post-process expects target_sizes in (height, width)
    target_sizes = [(pil_image.height, pil_image.width)]
    results = feature_extractor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=target_sizes
    )[0]

    bill_count = 0
    coin_count = 0
    annotated_image = image.copy()

    # Process detections and annotate image
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        score = score.cpu().item()
        label = label.cpu().item()
        box = box.cpu().numpy().tolist()

        # Skip label 0 if it's a placeholder
        if label == 0:
            continue

        text = f"{ID_TO_NAME.get(label, 'unknown')} {score:.2f}"
        # Determine color: blue for bills (if "eur" in label), green for coins
        if "eur" in ID_TO_NAME.get(label, "").lower():
            color = (255, 0, 0)
            bill_count += 1
        else:
            color = (0, 255, 0)
            coin_count += 1

        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Calculate text size and background for better readability
        box_height = y_max - y_min
        font_scale = box_height / 200.0
        thickness = max(1, int(font_scale * 2))
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(annotated_image, (x_min, y_min), (x_min + text_w, y_min + text_h + baseline), (255, 255, 255), thickness=-1)
        cv2.putText(annotated_image, text, (x_min, y_min + text_h), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

    return annotated_image, bill_count, coin_count
