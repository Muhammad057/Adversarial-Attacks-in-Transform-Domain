import os
import glob
import shutil
import cv2 as cv
import numpy as np
from methods.transform import TransformDomain
from utils.compute_dwt import calculate_DWT, calculate_IDWT
from utils.visualizer import Visualizer
from utils import settings as c

class DWT(TransformDomain):
    def __init__(self, technique):
        super().__init__(technique)

        self.wavelet_family = c.WAVELET_FAMILY
        self.image_size = c.IMAGE_SIZE
        self.save_perturbations = c.SAVE_PERTURBATIONS
        self.save_adversarial_examples = c.SAVE_ADVERSARIAL_EXAMPLES
        self.save_host_image_dwt_info = c.SAVE_HOST_IMAGE_DWT_INFO
        self.save_stego_image = c.SAVE_STEGO_IMAGE
        self.file_name_dwt_info_host_image = c.FILENAME_HOST_IMAGE_DWT_INFO
        self.file_name_dwt_info_water_mark_image = c.FILENAME_WATERMARK_IMAGE_DWT_INFO
        self.file_name_dwt_stego_image = c.FILENAME_DWT_STEGO_IMAGE
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

    def generate_dwt_water_mark_image(self):
        if self.is_exist_water_mark:
            watermark_image = cv.imread(self.water_mark_image)
            watermark_image = cv.resize(watermark_image, (self.image_size, self.image_size))
            watermark_image = watermark_image.astype(np.float32)
            blue_water_mark, green_water_mark, red_water_mark = cv.split(watermark_image)

            self.blue_water_mark_dwt_info = calculate_DWT(blue_water_mark, self.wavelet_family)
            self.blue_water_mark_LL, (
                self.blue_water_mark_LH, self.blue_water_mark_HL, self.blue_water_mark_HH
                ) = self.blue_water_mark_dwt_info

            self.green_water_mark_dwt_info = calculate_DWT(green_water_mark, self.wavelet_family)
            self.green_water_mark_LL, (
                self.green_water_mark_LH, self.green_water_mark_HL, self.green_water_mark_HH
                ) = self.green_water_mark_dwt_info
            
            self.red_water_mark_dwt_info = calculate_DWT(red_water_mark, self.wavelet_family)
            self.red_water_mark_LL, (
                self.red_water_mark_LH, self.red_water_mark_HL, self.red_water_mark_HH
                ) = self.red_water_mark_dwt_info

            
            if self.save_host_image_dwt_info:
                self.dummy = [
                    self.blue_water_mark_dwt_info,
                    self.green_water_mark_dwt_info,
                    self.red_water_mark_dwt_info
                    ]
                
                Visualizer(
                    self.water_mark_image, self.file_name_dwt_info_water_mark_image, 
                    self.perturbations_dir, self.dummy
                ).visualize_dwt_data()

    def generate_stego_image(self):
        if self.is_exist_host_image_dir:
            host_image_files = glob.glob(self.host_image_dir + '\*.jpeg')

            for self.file in host_image_files:
                host_image = cv.imread(self.file)
                host_image = cv.resize(host_image, (self.image_size, self.image_size))
                host_image = host_image.astype(np.float32) 
                blue_host, green_host, red_host = cv.split(host_image)

                blue_host_dwt_info = calculate_DWT(blue_host, self.wavelet_family)
                self.blue_host_LL, (
                    self.blue_host_LH, self.blue_host_HL, self.blue_host_HH
                    ) = blue_host_dwt_info

                green_host_dwt_info = calculate_DWT(green_host, self.wavelet_family)
                self.green_host_LL, (
                    self.green_host_LH, self.green_host_HL, self.green_host_HH
                    ) = green_host_dwt_info

                red_host_dwt_info = calculate_DWT(red_host, self.wavelet_family)
                self.red_host_LL, (
                    self.red_host_LH, self.red_host_HL, self.red_host_HH
                    ) = red_host_dwt_info

                self.blue_LL = self.blue_host_LL
                self.blue_LH = (1 - c.FACTOR_DWT) * self.blue_host_LH + c.FACTOR_DWT * self.blue_water_mark_LH
                self.blue_HL = (1 - c.FACTOR_DWT) * self.blue_host_HL + c.FACTOR_DWT * self.blue_water_mark_HL
                self.blue_HH = (1 - c.FACTOR_DWT) * self.blue_host_HH + c.FACTOR_DWT * self.blue_water_mark_HH

                self.green_LL = self.green_host_LL
                self.green_LH = (1 - c.FACTOR_DWT) * self.green_host_LH + c.FACTOR_DWT * self.green_water_mark_LH
                self.green_HL = (1 - c.FACTOR_DWT) * self.green_host_HL + c.FACTOR_DWT * self.green_water_mark_HL
                self.green_HH = (1 - c.FACTOR_DWT) * self.green_host_HH + c.FACTOR_DWT * self.green_water_mark_HH

                self.red_LL = self.red_host_LL
                self.red_LH = (1 - c.FACTOR_DWT) * self.red_host_LH + c.FACTOR_DWT * self.red_water_mark_LH
                self.red_HL = (1 - c.FACTOR_DWT) * self.red_host_HL + c.FACTOR_DWT * self.red_water_mark_HL
                self.red_HH = (1 - c.FACTOR_DWT) * self.red_host_HH + c.FACTOR_DWT * self.red_water_mark_HH

                self.blue_stego_image = calculate_IDWT(self.blue_LL, self.blue_LH, self.blue_HL, self.blue_HH, c.WAVELET_FAMILY)
                self.green_stego_image = calculate_IDWT(self.green_LL, self.green_LH, self.green_HL, self.green_HH, c.WAVELET_FAMILY)
                self.red_stego_image = calculate_IDWT(self.red_LL, self.red_LH, self.red_HL, self.red_HH, c.WAVELET_FAMILY)

                self.stego_image = cv.merge((self.blue_stego_image, self.green_stego_image, self.red_stego_image))


                if self.save_stego_image:
                    Visualizer(
                        self.file, self.file_name_dwt_stego_image,
                        self.adversarial_examples_dir, self.stego_image
                        ).visualize_data()

        else:
            print("Error - host image directory is empty: %s" % self.host_image_dir)



