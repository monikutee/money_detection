from annotate_image import run_inference
import cv2


# Load image using OpenCV and convert from BGR to RGB
image_file = "test.jpg"
image = cv2.imread(image_file)
if image is None:
    raise ValueError(f"Could not read image {image_file}")

annotated_image, bill_count, coin_count = run_inference(image)

# Print stats
print("Inference complete.")
print(f"Total bills detected: {bill_count}")
print(f"Total coins detected: {coin_count}")

# Save the annotated image
output_filename = "annotated_image.jpg"
if cv2.imwrite(output_filename, annotated_image):
    print(f"Annotated image saved as {output_filename}")
else:
    print("Error: Could not save the annotated image.")