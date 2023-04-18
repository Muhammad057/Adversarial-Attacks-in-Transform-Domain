from methods.dct import DCT
from methods.fft import FFT
from methods.dwt import DWT

from utils import settings as c

if __name__ == '__main__':

    if c.GENERATE_ADVERSARIAL_EXAMPLES_DCT:
        sampler = DCT("running DCT")
        print (sampler.get_technique_name())
        sampler.generate_masks_water_mark()
        sampler.generate_masks_host()
        sampler.generate_dct_water_mark_image()
        sampler.generate_stego_image()

    if c.GENERATE_ADVERSARIAL_EXAMPLES_FFT:
        sampler = FFT("running FFT")
        print (sampler.get_technique_name())
        sampler.create_output_directory()
        sampler.generate_fft_water_mark_image()
        sampler.generate_stego_image()
    
    if c.GENERATE_ADVERSARIAL_EXAMPLES_DWT:
        sampler = DWT("running DWT")
        print (sampler.get_technique_name())
        sampler.create_output_directory()
        sampler.generate_dwt_water_mark_image()
        sampler.generate_stego_image()
    
