import easyocr
import cv2
# Initialize EasyOCR reader (English only, add languages as needed)
reader = easyocr.Reader(['en'])

def recognize_text(img, region):
    """
    Recognizes text within a specified region of the image.
    Args:
        img (numpy.ndarray): Grayscale or color image.
        region (dict): Dictionary with 'bbox': (x, y, w, h)
    Returns:
        str: Recognized text in the region.
    """
    x, y, w, h = region["bbox"]
    # Crop the region from the image
    crop = img[y:y+h, x:x+w]
    # If grayscale, convert to RGB for EasyOCR
    if len(crop.shape) == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    # Run OCR
    result = reader.readtext(crop, detail=0)
    return " ".join(result)  # Join multiple lines of text if detected