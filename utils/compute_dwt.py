import pywt
import numpy as np

def calculate_DWT(image, wavelet_name) -> tuple:
    coeffs2 = pywt.dwt2(image, wavelet_name)
    return coeffs2

def calculate_IDWT(
    image_LL, image_LH,
    image_HL, image_HH, wavelet_name
    ) -> np.ndarray:
    
    return pywt.idwt2(
        [image_LL, (image_LH, image_HL, image_HH)],
        wavelet_name)

