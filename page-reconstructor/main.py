import os
import matplotlib.pyplot as plt
from preprocess.cleaner import (
    reduce_noise,
    binarize_image,
    resize_image,
    deskew_image,
    sharpen_image
)
from layout_detection.detect_blocks import extract_layout
from text_recognition.ocr_engine import recognize_text
from renderer.pdf_generator import generate_output

def show_image(img, title):
    plt.figure()
    if len(img.shape) == 2:  # Grayscale
        plt.imshow(img, cmap='gray')
    else:  # Color
        plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

def main(image_name, output_path="output.pdf"):
    # --- DEBUG ---
    print("[INFO] Starting page reconstruction...")
    print(f"[DEBUG] Image path set to: {image_path}")

    # Step 1: Clean image
    cleaned_img = reduce_noise(image_path)
    # --- DEBUG ---
    print("[DEBUG] Image cleaning complete.")
    # show_image(cleaned_img, "Cleaned Image")

# Step 2: Binarize image
    binarized_img = binarize_image(cleaned_img)
    print("[DEBUG] Binarization complete.")
    show_image(binarized_img, "Binarized Image")

    # Step 3: Resize image
    resized_img = resize_image(binarized_img, size=(1024, 1024))
    print("[DEBUG] Resizing complete.")
    show_image(resized_img, "Resized Image")

    # Step 4: Deskew image
    deskewed_img = deskew_image(resized_img)
    print("[DEBUG] Deskewing complete.")
    show_image(deskewed_img, "Deskewed Image")

    # Step 5: Sharpen image
    sharpened_img = sharpen_image(deskewed_img)
    print("[DEBUG] Sharpening complete.")
    show_image(sharpened_img, "Sharpened Image")

    # Step 6: Detect layout/blocks
    layout_data = extract_layout(cleaned_img)
    # --- DEBUG ---
    print("[DEBUG] Layout detection complete.")
    # if "layout_img" in layout_data:
    #     show_image(layout_data["layout_img"], "Detected Layout")
    # else:
    #     show_image(cleaned_img, "Detected Layout (No layout_img provided)")

    # Step 7: OCR for each detected block
    for region in layout_data["text_regions"]:
        region["text"] = recognize_text(cleaned_img, region)
    # --- DEBUG ---
    print("[DEBUG] OCR complete for all detected blocks.")
    # show_image(cleaned_img, "OCR Regions (after text extraction)")

    # Step 8: Render output (PDF or other format)
    print(f"[DEBUG] Generating output PDF.")
    generate_output(layout_data, output_path)
    # --- DEBUG ---
    print("[DEBUG] Output rendering complete.")

    print(f"[SUCCESS] Output saved to {output_path}")

if __name__ == "__main__":
    # Get the absolute path to the page-reconstructor directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    image_name = input("Enter image filename (in assets/test_images): ")
    output_name = input("Enter output path [default: output.pdf]: ") or "output.pdf"
    
    # Input and output paths relative to page-reconstructor
    image_path = os.path.join(base_dir, "assets", "test_images", image_name)
    output_path = os.path.join(base_dir, "outputs", "pdfs", output_name)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    main(image_name, output_path)
    print("[INFO] Page reconstruction completed successfully.")