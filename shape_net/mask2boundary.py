# functions to convert binary mask images to boundaryies (files&images)
# zzhu, 2020/03/22
import os
import glob
import cv2

import numpy as np

def extract_contours(img):
    '''
    Extract the contours of a binary mask image, using the opencv module findContours
    img: 1-channel binary mask image
    return a bounrary/edge/contour image and a vector of the contour points
    OpenCV version >= 3.2
    Zhe Zhu, 2020/03/22
    '''
    img2, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt_img = np.zeros(img.shape,dtype=np.uint8)
    for contour in contours:
        for pts_arr in contour: # super inefficient, not recommended
            pts = pts_arr[0]
            cnt_img[pts[1],pts[0]] = 255
    return cnt_img, contours


def test():
    '''
    For test use only
    '''
    sample_img_file = '/mnt/sdc/ShapeTexture/simulation_data/0224/val/val_00003_mask.png'
    output_img_file = '/mnt/sdc/code/shape_segmentation/shape_net/test.png'
    mask = cv2.imread(sample_img_file,cv2.IMREAD_GRAYSCALE)
    contour_img, contours = extract_contours(mask)
    cv2.imwrite(output_img_file,contour_img)
    print(len(contours))

if __name__=="__main__":
    test()
