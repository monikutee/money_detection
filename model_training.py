import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from transformers import (
    AutoFeatureExtractor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer
)

# Custom Dataset class for local COCO data.
class LocalCocoDataset(Dataset):
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
        # Add "class_labels" to each annotation so that the loss function finds it.
        for ann in annotations:
            ann["class_labels"] = ann["category_id"]
        # Wrap annotations in the expected dict format.
        annotation_dict = {"image_id": image_id, "annotations": annotations}
        inputs = self.feature_extractor(images=image, annotations=annotation_dict, return_tensors="pt")
        # Squeeze the batch dimension for tensors only.
        inputs = {k: (v.squeeze(0) if hasattr(v, "squeeze") else v) for k, v in inputs.items()}
        return inputs

# Updated collate function: flatten the "labels" list.
def collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        if key == "labels":
            # Each item[key] is a list of dicts. Flatten them.
            collated[key] = [d for item in batch for d in item[key]]
        elif hasattr(batch[0][key], "unsqueeze"):
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            collated[key] = [item[key] for item in batch]
    return collated

def main():
    # Update these paths to match your dataset structure.
    base_dir = "./bills_and_coins.v3i.coco"
    train_images_dir = os.path.join(base_dir, "train")
    train_annotation_file = os.path.join(base_dir, "train", "_annotations.coco.json")
    
    valid_images_dir = os.path.join(base_dir, "valid")
    valid_annotation_file = os.path.join(base_dir, "valid", "_annotations.coco.json")
    
    output_dir = "./yolos_results_v2"

    # Load the feature extractor and pretrained YOLOS model.
    feature_extractor = AutoFeatureExtractor.from_pretrained("hustvl/yolos-small")
    model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-small")

    train_dataset = LocalCocoDataset(train_images_dir, train_annotation_file, feature_extractor)
    valid_dataset = LocalCocoDataset(valid_images_dir, valid_annotation_file, feature_extractor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,  # Number of epochs (can increase if needed)
        per_device_train_batch_size=4,  # Increase batch size if possible
        per_device_eval_batch_size=4,
        eval_strategy="epoch",
        logging_steps=50,
        save_strategy="epoch",
        remove_unused_columns=False,
        
        # ✅ Set custom learning rate
        learning_rate=5e-5,  # Default is 5e-5; try reducing to 1e-5 if training is unstable
        warmup_ratio=0.1,  # Warmup helps stabilize early training
        weight_decay=0.01,  # Regularization to prevent overfitting
        gradient_accumulation_steps=2,  # If GPU memory is limited, accumulate gradients
        max_grad_norm=1.0,  # Prevent exploding gradients
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
    )

    trainer.train()
    
    save_path = "./my_trained_yolos_model_v2"
    model.save_pretrained(save_path)
    feature_extractor.save_pretrained(save_path)
    print(f"Training complete. Model saved to '{save_path}'.")

if __name__ == "__main__":
    main()
