import numpy as np
import cv2
import math
import face_alignment
import os
from multiprocessing import Pool
import time

def align_crop_img(input_img,mask, lmks,size=256):
    
    img_h,img_w = input_img.shape[0],input_img.shape[1]
    
    center = ( img_w / 2.0, img_h / 2.0)
    lm_eye_left      = lmks[36 : 42]  # left-clockwise
    lm_eye_right     = lmks[42 : 48]  # left-clockwise

    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    angle = np.arctan2((eye_right[1] - eye_left[1]), (eye_right[0] - eye_left[0])) / np.pi * 180

    RotateMatrix = cv2.getRotationMatrix2D(center, angle, scale=1)
    
    
    rotated_img = cv2.warpAffine(input_img, RotateMatrix, (size,size))
    mask = cv2.warpAffine(mask, RotateMatrix, (size,size))
    return rotated_img,mask
    


def draw_lmk(img,lmk,scale=10):
    draw_img = img.copy()
    for p in lmk:
        cv2.circle(draw_img,(int(p[0]),int(p[1])),scale,[0,255,0])
    return draw_img 

def run(img_paths):
   
    lmk_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
    bbox = [114,114,398,398]
    k = 1
    for img_path in img_paths:
        img = cv2.imread(img_path)
        mask_path = img_path.replace('img','mask')
        mask = cv2.imread(mask_path)
        landmarks = lmk_detector.get_landmarks_from_image(img[...,::-1], [bbox])[0]
       
        img,mask = align_crop_img(img,mask,landmarks[:,:2],size=512)
        
        save_img_path = img_path.replace('process','align')
        save_mask_path = mask_path.replace('process','align')
        os.makedirs(os.path.split(save_img_path)[0],exist_ok=True)
        os.makedirs(os.path.split(save_mask_path)[0],exist_ok=True)

        cv2.imwrite(save_img_path,img)
        cv2.imwrite(save_mask_path,mask)
        print('\rhave done %06d'%k,end='',flush=True)
        k += 1
    print()

if __name__ == "__main__":
    base = '../dataset/process/img'
    id_paths = [os.path.join(base,f) for f in os.listdir(base)]
    video_paths = [os.path.join(id_base,f) for id_base in id_paths for f in os.listdir(id_base)]
    img_paths = [os.path.join(img_base,f) for img_base in video_paths for f in os.listdir(img_base)]
    pool_num = 3
    length = len(img_paths)
    
    dis = math.ceil(length/float(pool_num))
    
    # work(video_paths[i*dis:(i+1)*dis],save_img_base,save_pkl_base,save_pair_base,i)
    t1 = time.time()
    print('***************all length: %d ******************'%length)
    p = Pool(pool_num)
    for i in range(pool_num):
        p.apply_async(run, args = (img_paths[i*dis:(i+1)*dis],))   
    p.close() 
    p.join()
    print("all the time: %s"%(time.time()-t1))