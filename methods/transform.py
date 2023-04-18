# from inspect import _Object
import os
# from methods.dct import DCT
from utils import settings as c

class TransformDomain(object):
        def __init__(self, technique):
            self.technique = technique
            self.is_exist_water_mark = os.path.exists(c.WATERMARK_IMAGE)
            self.is_exist_host_image_dir = os.path.exists(c.HOST_IMAGE_DIR)
        
        def get_technique_name(self):
            return self.technique


# sampler = DCT("running DCT")
