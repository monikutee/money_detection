import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection

def evaluate_model(annotation_file, prediction_file):
    coco_gt = COCO(annotation_file)  # Ground truth annotations
    coco_dt = coco_gt.loadRes(prediction_file)  # Model predictions

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()  # Prints mAP and other metrics

# Example usage:
evaluate_model("bills_and_coins.v3i.coco/test/_annotations.coco.json", "predictions.json")