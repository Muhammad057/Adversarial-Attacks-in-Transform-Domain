import scipy
import numpy as np
from numpy import r_
from scipy import fftpack

def dct2(block):
    return scipy.fftpack.dct(scipy.fftpack.dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(block):
    return scipy.fftpack.idct(scipy.fftpack.idct(block, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def calculate_IDCT(idct_img):
    """_summary_

    Args:
        idct_img (_type_): _description_

    Returns:
        _type_: _description_
    """
    imsize = idct_img.shape
    im_dct = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            im_dct[i:(i+8), j:(j+8)] = idct2(idct_img[i:(i+8), j:(j+8)])
    return im_dct

def calculate_DCT(dct_img, mask_dc, mask_edges):
    """
    Do 64x64 DCT on image
    """
    imsize = dct_img.shape
    dc_info = np.zeros(imsize)
    edge_info = np.zeros(imsize)

    if mask_dc is None and mask_edges is not None:
        dc_info = None

        for i in r_[:imsize[0]:8]:
            for j in r_[:imsize[1]:8]:
                    edge_info[i:(i+8), j:(j+8)] = dct2(dct_img[i:(i+8), j:(j+8)]) * mask_edges

    elif mask_dc is not None and mask_edges is None:
        edge_info = None

        for i in r_[:imsize[0]:8]:
            for j in r_[:imsize[1]:8]:
                    dc_info[i:(i+8), j:(j+8)] = dct2(dct_img[i:(i+8), j:(j+8)]) * mask_dc


    elif mask_dc is not None and mask_edges is not None:
        for i in r_[:imsize[0]:8]:
            for j in r_[:imsize[1]:8]:
                    dc_info[i:(i+8), j:(j+8)] = dct2(dct_img[i:(i+8), j:(j+8)]) * mask_dc
                    edge_info[i:(i+8), j:(j+8)] = dct2(dct_img[i:(i+8), j:(j+8)]) * mask_edges
    
    else:
        dc_info = None
        edge_info = None
    
    return dc_info, edge_info