
import cv2
import numpy as np
from PIL import Image


def enhance_kidney_image(image, size=(224, 224)):
    """
    Experimental fuzzy-membership contrast enhancement.
    Not a verified reproduction of the STREAMLINERS method.
    """

    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    image = np.asarray(image)

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("Expected a grayscale or RGB image.")

    gray = cv2.resize(gray, size)
    gray = gray.astype(np.float32) / 255.0

    # Convert pixel intensities into fuzzy memberships.
    membership = np.clip(gray, 0.0, 1.0)

    # Nonlinear membership transformation.
    enhanced = np.where(
        membership <= 0.5,
        2.0 * membership**2,
        1.0 - 2.0 * (1.0 - membership)**2
    )

    enhanced = np.clip(enhanced * 255.0, 0, 255)
    enhanced = enhanced.astype(np.uint8)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
