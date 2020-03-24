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

def cvt_mask2cnt_single(mask_img_file, cnt_img_file):
    '''
    Convert a binary mask image to a corresponding contour image
    mask_img_file: source image file to load
    cnt_img_file: target image file to write
    '''
    mask_img = cv2.imread(mask_img_file,cv2.IMREAD_GRAYSCALE)
    cnt_img, cnts = extract_contours(mask_img)
    cv2.imwrite(cnt_img_file,cnt_img)

def cvt_mask2cnt_batch(mask_file_list,cnt_file_list):
    '''
    Batch convert mask images to contour images
    mask_file_list: source mask image files
    cnt_file_list: target contour image files
    '''
    img_num = len(mask_file_list)
    for i in range(img_num):
        src_img_file = mask_file_list[i]
        tgt_img_file = cnt_file_list[i]
        cvt_mask2cnt_single(src_img_file,tgt_img_file)

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

def test_1():
    '''
    Debug use. To solve the problem that still output mask image
    2020/03/23
    '''
    src_img_file = '/mnt/sdc/ShapeTexture/simulation_data/0224/val/val_00003.png'
    tgt_img_file = '/mnt/sdc/ShapeTexture/simulation_data/debug/20200323.png'
    cvt_mask2cnt_single(src_img_file,tgt_img_file)

def task_1():
    '''
    Convert the 0219, 0224 masks to contours
    '''
    src_folder_list = ['/mnt/sdc/ShapeTexture/simulation_data/0219/train',
                       '/mnt/sdc/ShapeTexture/simulation_data/0219/val',
                       '/mnt/sdc/ShapeTexture/simulation_data/0224/train',
                       '/mnt/sdc/ShapeTexture/simulation_data/0224/val']
    tgt_folder_list = ['/mnt/sdc/ShapeTexture/simulation_data/0219_contour/train',
                       '/mnt/sdc/ShapeTexture/simulation_data/0219_contour/val',
                       '/mnt/sdc/ShapeTexture/simulation_data/0224_contour/train',
                       '/mnt/sdc/ShapeTexture/simulation_data/0224_contour/val']
    folder_num = len(src_folder_list)
    for i in range(folder_num):
        src_folder = src_folder_list[i]
        tgt_folder = tgt_folder_list[i]
        src_mask_file_list = glob.glob(src_folder+'/*mask.png')
        print("src_folder: {0}".format(src_folder))
        print("tgt_folder: {0}".format(tgt_folder))
        tgt_cnt_file_list = []
        for src_mask_file in src_mask_file_list:
            mask_file_name = os.path.basename(src_mask_file)
            cnt_file_name = os.path.join(tgt_folder,mask_file_name)
            tgt_cnt_file_list.append(cnt_file_name)
        print("Converting {0} images".format(len(tgt_cnt_file_list)))
        cvt_mask2cnt_batch(src_mask_file_list,tgt_cnt_file_list)


if __name__=="__main__":
    #test()
    #test_1()
    task_1()
