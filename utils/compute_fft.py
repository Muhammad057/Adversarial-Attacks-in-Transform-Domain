import numpy as np

def calculate_FFT(image, magnitude_flag, phase_flag):
    image_fft = np.fft.fft2(image)
    image_shift = np.fft.fftshift(image_fft)

    if magnitude_flag:
        magnitude = np.abs(image_shift)
    else:
        magnitude = None    

    if phase_flag:
        phase = np.exp(1j * np.angle(image_shift))
    else:
        phase = None
       
    return magnitude, phase 

def calculate_IFFT(factor, host_magnitude, host_phase, water_mark_phase):
    combine_phase = (1 - factor) * host_phase + factor * water_mark_phase
    combine =  np.multiply(np.abs(host_magnitude), combine_phase)
    ifft = np.abs(np.fft.ifft2(combine))
    return ifft

