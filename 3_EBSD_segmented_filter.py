import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import label, regionprops

def filter_small_regions (Im,
                          region_size_min = 10) :
    """ Extract datas grain per grain from im1.
    
    Args:
       Im : np.array
           Microstructure skeleton (black and white datas (0 or 255))
       region_size_min : int, optional
           Number minimal of pixel per region to keep.
           
    Returns:
       Im : list
           List of skimage regions
    """
    binary_microstructure = Im < 250
    label_img=label(binary_microstructure)
    all_regions = regionprops(label_img)
    regions = []
    coords = []
    for region in (all_regions):
        if region.area > region_size_min :
            regions.append(region)
            coords.append(region.coords)
    
    mask = np.zeros_like(Im, dtype=np.uint8)
    for coord in coords :
        mask[coord[:, 0], coord[:, 1]] = 1
        
    return(mask)

if __name__ == '__main__' :
    EBSD_matlab_map = cv2.imread('3_EBSD_presegmented_bw.png', 0)
    EB1 = EBSD_matlab_map[98:108,165:180]
    EB2 = EBSD_matlab_map[170:187,191:208]
    
    mask = filter_small_regions(EBSD_matlab_map)
    np.save('3_Filtered_EBSD_skeleton.npy', mask)