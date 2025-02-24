from annotate_image import run_inference
import cv2
import argparse


def main(image_file):
    image = cv2.imread(image_file)
    if image is None:
        image = cv2.imread('test.jpg')

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Eur atpazinimas")
    parser.add_argument("--image", type=str, required=False, help="Paveiksliuko kelias")
    args = parser.parse_args()
    main(args.image)
