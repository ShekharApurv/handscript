import cv2
import numpy as np
from PIL import Image

def reduce_noise(input_path, output_path=None):
    """
    Reads an image, reduces noise, and saves or returns the cleaned image.
    Args:
        input_path (str): Path to the input image.
        output_path (str, optional): Path to save the cleaned image.
    Returns:
        cleaned_img (numpy.ndarray): Denoised image.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {input_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    if output_path:
        cv2.imwrite(output_path, denoised)
    return denoised

def binarize_image(image_array, output_path=None):
    """
    Applies Otsu's thresholding to binarize an image.
    Args:
        image_array (numpy.ndarray): The input grayscale (denoised) image array.
        output_path (str, optional): Path to save the binarized image.
    Returns:
        binarized_img (numpy.ndarray): Binarized image.
    """
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if output_path:
        cv2.imwrite(output_path, binarized)
    return binarized

def resize_image(image_array, size=(1024, 1024), output_path=None):
    """
    Resizes the image to the given size.
    Args:
        image_array (numpy.ndarray): Input image array.
        size (tuple): Desired size (width, height).
        output_path (str, optional): Path to save the resized image.
    Returns:
        resized_img (numpy.ndarray): Resized image.
    """
    resized = cv2.resize(image_array, size, interpolation=cv2.INTER_AREA)
    if output_path:
        cv2.imwrite(output_path, resized)
    return resized

def deskew_image(image_array, output_path=None):
    """
    Deskews the image using moments.
    Args:
        image_array (numpy.ndarray): Input image array.
        output_path (str, optional): Path to save the deskewed image.
    Returns:
        deskewed_img (numpy.ndarray): Deskewed image.
    """
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if output_path:
        cv2.imwrite(output_path, deskewed)
    return deskewed

def sharpen_image(image_array, output_path=None):
    """
    Sharpens the image using a kernel.
    Args:
        image_array (numpy.ndarray): Input image array.
        output_path (str, optional): Path to save the sharpened image.
    Returns:
        sharpened_img (numpy.ndarray): Sharpened image.
    """
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(image_array, -1, kernel)
    if output_path:
        cv2.imwrite(output_path, sharpened)
    return sharpened

# Example usage:
# cleaned = reduce_noise("input.jpg", "output_cleaned.jpg")
# binarized_image_array = binarize_image(cleaned_image_array, "test2_binarized.jpg")