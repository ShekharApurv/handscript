import cv2
import numpy as np

def extract_layout(img):
    """
    Detects text blocks in the image and returns their bounding boxes.
    Args:
        img (numpy.ndarray): Grayscale or color image.
    Returns:
        dict: {
            "text_regions": [ {"bbox": (x, y, w, h)} ],
            "layout_img": image with detected blocks drawn
        }
    """
    # Ensure grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Thresholding to get binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate to merge text into blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_regions = []
    layout_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter small regions
        if w > 30 and h > 10:
            text_regions.append({"bbox": (x, y, w, h)})
            cv2.rectangle(layout_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return {
        "text_regions": text_regions,
        "layout_img": layout_img
    }