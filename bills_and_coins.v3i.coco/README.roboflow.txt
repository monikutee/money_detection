
bills_and_coins - v3 2025-02-23 4:53pm
==============================

This dataset was exported via roboflow.com on February 23, 2025 at 2:54 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 3863 images.
Eur-1-2-5-10-20-50-100-200 are annotated in COCO format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 1000x400 (Stretch)
* Auto-contrast via contrast stretching

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Randomly crop between 0 and 22 percent of the image
* Random rotation of between -45 and +45 degrees
* Random shear of between -11° to +11° horizontally and -11° to +11° vertically
* Random exposure adjustment of between -12 and +12 percent
* Salt and pepper noise was applied to 0.97 percent of pixels


