import imp
import os
import glob
import shutil
import cv2 as cv
import numpy as np
from methods.transform import TransformDomain
from utils.compute_fft import calculate_FFT, calculate_IFFT
from utils.visualizer import Visualizer
from utils import settings as c

class FFT(TransformDomain):
    def __init__(self, technique):
        super().__init__(technique)

        self.save_stego_image = c.SAVE_STEGO_IMAGE
        self.factor_fft = c.FACTOR_FFT
        self.magnitude_host = c.MAGNITUDE_HOST
        self.phase_host = c.PHASE_HOST
        self.magnitude_water_mark = c.MAGNITUDE_WATER_MARK
        self.phase_water_mark = c.PHASE_WATER_MARK
        self.image_size = c.IMAGE_SIZE
        self.save_perturbations = c.SAVE_PERTURBATIONS
        self.save_adversarial_examples = c.SAVE_ADVERSARIAL_EXAMPLES
        self.filename_fft_stego_image = c.FILENAME_FFT_STEGO_IMAGE
        self.perturbations_dir = c.PERTURBATIONS_DIR
        self.adversarial_examples_dir = c.ADVERSARIAL_EXAMPLES_DIR
        self.water_mark_image = c.WATERMARK_IMAGE
        self.host_image_dir = c.HOST_IMAGE_DIR
        
    def create_output_directory(self):
        if self.save_perturbations:
            IS_EXIST_PERTURBATION_PATH = os.path.exists(self.perturbations_dir)
        
            if IS_EXIST_PERTURBATION_PATH:
                try:
                    shutil.rmtree(self.perturbations_dir)
                except OSError as e:
                    print("Error: %s : %s" % (self.perturbations_dir), e.strerror)

            os.makedirs(self.perturbations_dir)

        if self.save_adversarial_examples:
            IS_EXIST_ADV_PATH = os.path.exists(self.adversarial_examples_dir)

            if IS_EXIST_ADV_PATH:
                try:
                    shutil.rmtree(self.adversarial_examples_dir)
                except OSError as e:
                    print("Error: %s : %s" % (self.adversarial_examples_dir), e.strerror)

            os.makedirs(self.adversarial_examples_dir)

    def generate_fft_water_mark_image(self):
        if self.is_exist_water_mark:
            watermark_image = cv.imread(self.water_mark_image)
            watermark_image = cv.resize(watermark_image, (self.image_size, self.image_size))
            watermark_image = watermark_image.astype(np.float32)
            blue_water_mark, green_water_mark, red_water_mark = cv.split(watermark_image)


            self.blue_water_mark_magnitude_info, self.blue_water_mark_phase_info = calculate_FFT(blue_water_mark,
                                                                                                self.magnitude_water_mark,
                                                                                                self.phase_water_mark)

            self.green_water_mark_magnitude_info, self.green_water_mark_phase_info = calculate_FFT(green_water_mark,
                                                                                                self.magnitude_water_mark,
                                                                                                self.phase_water_mark)

            self.red_water_mark_magnitude_info, self.red_water_mark_phase_info = calculate_FFT(red_water_mark,
                                                                                                self.magnitude_water_mark,
                                                                                                self.phase_water_mark)

    def generate_stego_image(self):
        if self.is_exist_host_image_dir:
            host_image_files = glob.glob(self.host_image_dir + '\*.jpeg')

            for self.file in host_image_files:
                host_image = cv.imread(self.file)
                host_image = cv.resize(host_image, (self.image_size, self.image_size))
                host_image = host_image.astype(np.float32) 
                blue_host, green_host, red_host = cv.split(host_image)

                self.blue_host_magnitude_info, self.blue_host_phase_info = calculate_FFT(blue_host,
                                                                                        self.magnitude_host,
                                                                                        self.phase_host)

                self.green_host_magnitude_info, self.green_host_phase_info = calculate_FFT(green_host,
                                                                                        self.magnitude_host,
                                                                                        self.phase_host)

                self.red_host_magnitude_info, self.red_host_phase_info = calculate_FFT(red_host,
                                                                                    self.magnitude_host,
                                                                                    self.phase_host)                

                self.ifft_blue = calculate_IFFT(self.factor_fft, self.blue_host_magnitude_info, self.blue_host_phase_info, self.blue_water_mark_phase_info)
                self.ifft_green = calculate_IFFT(self.factor_fft, self.green_host_magnitude_info, self.green_host_phase_info, self.green_water_mark_phase_info)
                self.ifft_red = calculate_IFFT(self.factor_fft, self.red_host_magnitude_info, self.red_host_phase_info, self.red_water_mark_phase_info)

                self.stego_image = cv.merge((self.ifft_blue, self.ifft_green, self.ifft_red))

                if self.save_stego_image:
                    print (self.stego_image)
                    Visualizer(self.file, self.filename_fft_stego_image, self.adversarial_examples_dir,
                        self.stego_image).visualize_data()
        else:
            print("Error - host image directory is empty: %s" % self.host_image_dir)


                

    