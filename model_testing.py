import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection

class LocalCocoTestDataset(Dataset):
    def __init__(self, images_dir, annotation_file, feature_extractor):
        self.images_dir = images_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.coco.imgs[image_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        # Add "class_labels" key for each annotation
        for ann in annotations:
            ann["class_labels"] = ann["category_id"]
        annotation_dict = {"image_id": image_id, "annotations": annotations}
        inputs = self.feature_extractor(images=image, annotations=annotation_dict, return_tensors="pt")
        # Remove the extra batch dimension if the value is a tensor.
        inputs = {k: (v.squeeze(0) if hasattr(v, "squeeze") else v) for k, v in inputs.items()}
        # Wrap the original image size in a list to preserve the (width, height) tuple.
        inputs["orig_size"] = [image.size]
        inputs["file_name"] = img_info["file_name"]
        inputs["image_id"] = image_id  # <-- Add image_id here
        return inputs

import json
from pycocotools.cocoeval import COCOeval

def test_model(dataset, model, feature_extractor, device, id_to_name, output_json="predictions.json"):
    model.to(device)
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    predictions = []
    
    for batch in dataloader:
        print(batch)
        pixel_values = batch["pixel_values"].to(device)
        orig_size = batch["orig_size"]  # list of tuples, each like (width, height)
        file_name = batch["file_name"]
        image_id = batch["image_id"].item()  # Extract image_id

        target_sizes = [(orig_size[0][1], orig_size[0][0])]
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        
        results = feature_extractor.post_process_object_detection(
            outputs, threshold=0.5, target_sizes=target_sizes
        )[0]

        # Convert results to COCO format
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x_min, y_min, x_max, y_max = box.tolist()
            width, height = x_max - x_min, y_max - y_min

            predictions.append({
                "image_id": image_id,
                "category_id": label.item(),
                "bbox": [x_min, y_min, width, height],
                "score": score.item()
            })

    # Save predictions to JSON file
    with open(output_json, "w") as f:
        json.dump(predictions, f)

    print(f"Predictions saved to {output_json}")

def main():
    base_dir = "./bills_and_coins.v3i.coco"
    test_images_dir = os.path.join(base_dir, "test")
    test_annotation_file = os.path.join(base_dir, "test", "_annotations.coco.json")
    
    model_dir = "./my_trained_yolos_model"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
    model = AutoModelForObjectDetection.from_pretrained(model_dir)
    
    test_dataset = LocalCocoTestDataset(test_images_dir, test_annotation_file, feature_extractor)
    
    # Build a mapping from category IDs to names using the COCO annotations.
    coco = COCO(test_annotation_file)
    id_to_name = {cat["id"]: cat["name"] for cat in coco.dataset["categories"]}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_model(test_dataset, model, feature_extractor, device, id_to_name)

if __name__ == "__main__":
    main()
