import os
import glob
import shutil
import cv2 as cv
import numpy as np
from methods.transform import TransformDomain
from utils.compute_dct import calculate_DCT, calculate_IDCT
from utils.visualizer import Visualizer
from utils import settings as c

class DCT(TransformDomain):
    def __init__(self, technique):
        super().__init__(technique)

        self.mask_size = c.BLOCK_SIZE_DCT
        self.factor_dct = c.FACTOR_DCT
        self.dc_mask_water_mark = c.MASK_WATER_MARK_DC
        self.edge_mask_water_mark = c.MASK_WATER_MARK_EDGES
        self.dc_mask_host = c.MASK_HOST_DC
        self.edge_mask_host = c.MASK_HOST_EDGES
        self.image_size = c.IMAGE_SIZE
        self.save_perturbations = c.SAVE_PERTURBATIONS
        self.save_adversarial_examples = c.SAVE_ADVERSARIAL_EXAMPLES
        self.save_dct_host_image_dc_info = c.SAVE_DCT_HOST_IMAGE_DC_INFO
        self.save_dct_host_image_edge_info = c.SAVE_DCT_HOST_IMAGE_EDGE_INFO
        self.save_stego_image = c.SAVE_STEGO_IMAGE
        self.file_name_dct_host_image_dc_info = c.FILENAME_DCT_HOST_IMAGE_DC_INFO
        self.file_name_dct_host_image_edge_info = c.FILENAME_DCT_HOST_IMAGE_EDGE_INFO
        self.file_name_dct_stego_image = c.FILENAME_DCT_STEGO_IMAGE
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

    def generate_masks_water_mark(self):
        if self.dc_mask_water_mark:
            self.mask_water_mark_dc = np.zeros((self.mask_size, self.mask_size))
            self.mask_water_mark_dc[0:1,0:1] = 1
        else:
            self.mask_water_mark_dc = None

        if self.edge_mask_water_mark:
            self.mask_water_mark_edges = np.ones((self.mask_size, self.mask_size))
            self.mask_water_mark_edges[0:1,0:1] = 0
        else:
            self.mask_water_mark_edges = None

    def generate_masks_host(self):
        if self.dc_mask_host:
            self.mask_host_dc = np.zeros((self.mask_size, self.mask_size))
            self.mask_host_dc[0:1,0:1] = 1
        else:
            self.mask_host_dc = None

        if self.edge_mask_host:
            self.mask_host_edges = np.ones((self.mask_size, self.mask_size))
            self.mask_host_edges[0:1,0:1] = 0
        else:
            self.mask_host_edges = None
 
    def generate_dct_water_mark_image(self):
        if self.is_exist_water_mark:
            watermark_image = cv.imread(self.water_mark_image)
            watermark_image = cv.resize(watermark_image, (self.image_size, self.image_size))
            watermark_image = watermark_image.astype(np.float32)
            blue_water_mark, green_water_mark, red_water_mark = cv.split(watermark_image)

            self.blue_water_mark_dc_info, self.blue_water_mark_edge_info = calculate_DCT(blue_water_mark, self.mask_water_mark_dc, self.mask_water_mark_edges)
            self.green_water_mark_dc_info, self.green_water_mark_edge_info = calculate_DCT(green_water_mark, self.mask_water_mark_dc, self.mask_water_mark_edges)
            self.red_water_mark_dc_info, self.red_water_mark_edge_info = calculate_DCT(red_water_mark, self.mask_water_mark_dc, self.mask_water_mark_edges)

            self.blue_water_mark_idct = calculate_IDCT(self.blue_water_mark_edge_info)
            self.green_water_mark_idct = calculate_IDCT(self.green_water_mark_edge_info)
            self.red_water_mark_idct = calculate_IDCT(self.red_water_mark_edge_info)

            self.watermark_image_edge_info = cv.merge((self.blue_water_mark_idct, self.green_water_mark_idct, self.red_water_mark_idct))
        else:
            print("Error - watermark image not found: %s" % self.water_mark_image)
    
    def generate_stego_image(self):
        if self.is_exist_host_image_dir:
            host_image_files = glob.glob(self.host_image_dir + '\*.jpeg')

            for self.file in host_image_files:
                host_image = cv.imread(self.file)
                host_image = cv.resize(host_image, (self.image_size, self.image_size))
                host_image = host_image.astype(np.float32) 
                blue_host, green_host, red_host = cv.split(host_image)

                self.blue_host_dc_info, self.blue_host_edge_info = calculate_DCT(blue_host, self.mask_host_dc, self.mask_host_edges)
                self.green_host_dc_info, self.green_host_edge_info = calculate_DCT(green_host, self.mask_host_dc, self.mask_host_edges)
                self.red_host_dc_info, self.red_host_edge_info = calculate_DCT(red_host, self.mask_host_dc, self.mask_host_edges)

                self.blue_host_dc_info_idct = calculate_IDCT(self.blue_host_dc_info)
                self.green_host_dc_info_idct = calculate_IDCT(self.green_host_dc_info)
                self.red_host_dc_info_idct = calculate_IDCT(self.red_host_dc_info)

                self.blue_host_edge_info_idct = calculate_IDCT(self.blue_host_edge_info)
                self.green_host_edge_info_idct = calculate_IDCT(self.green_host_edge_info)
                self.red_host_edge_info_idct = calculate_IDCT(self.red_host_edge_info)

                self.host_image_dc_info = cv.merge((self.blue_host_dc_info_idct, self.green_host_dc_info_idct, self.red_host_dc_info_idct))
                self.host_image_edge_info = cv.merge((self.blue_host_edge_info_idct, self.green_host_edge_info_idct, self.red_host_edge_info_idct))

                self.stego_image = self.host_image_dc_info + (1 - self.factor_dct) * self.host_image_edge_info + self.factor_dct * self.watermark_image_edge_info

                if self.save_dct_host_image_dc_info:
                    Visualizer(
                        self.file, self.file_name_dct_host_image_dc_info, self.perturbations_dir,
                        self.host_image_dc_info
                        ).visualize_data()

                if self.save_dct_host_image_edge_info:
                    Visualizer(
                        self.file, self.file_name_dct_host_image_edge_info, self.perturbations_dir,
                        self.host_image_edge_info
                        ).visualize_data()

                if self.save_stego_image:
                    Visualizer(
                        self.file, self.file_name_dct_stego_image, self.adversarial_examples_dir,
                        self.stego_image
                        ).visualize_data()
        else:
            print("Error - host image directory is empty: %s" % self.host_image_dir)
