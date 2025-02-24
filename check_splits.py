import os
import json

def check_split(images_dir, annotation_file):
    # Read the annotation file.
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    # Extract file names from the annotation file.
    annotated_files = {img["file_name"] for img in data["images"]}

    # Get file names present in the images directory.
    actual_files = set(os.listdir(images_dir))

    # Find any missing or extra images.
    missing = annotated_files - actual_files
    extra = actual_files - annotated_files

    print(f"Checking {images_dir}")
    print(f"Total images in annotation: {len(annotated_files)}")
    print(f"Total images in folder: {len(actual_files)}")
    if missing:
        print("Missing images (listed in annotations but not found in folder):")
        for file in missing:
            print("  ", file)
    else:
        print("All annotated images are found in the folder.")
    if extra:
        print("Extra images (found in folder but not annotated):")
        for file in extra:
            print("  ", file)
    else:
        print("No extra images in the folder.")

# Update these paths to your local directories and files.
train_images_dir = "./bills_and_coins.v3i.coco/train"
train_annotation_file = os.path.join(train_images_dir, "_annotations.coco.json")

valid_images_dir = "./bills_and_coins.v3i.coco/valid"
valid_annotation_file = os.path.join(valid_images_dir, "_annotations.coco.json")

print("Training Split:")
check_split(train_images_dir, train_annotation_file)
print("\nValidation Split:")
check_split(valid_images_dir, valid_annotation_file)