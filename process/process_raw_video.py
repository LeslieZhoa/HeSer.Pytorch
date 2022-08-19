import os 
import imageio
import numpy as np
from face_alignment.detection.sfd import FaceDetector
import torch
import math
import cv2
import pdb
from multiprocessing import Process
import multiprocessing as mp

def get_video_info(base,save_base,q):
    for idname in os.listdir(base):
        idpath = os.path.join(base,idname)
        save_path = os.path.join(save_base,idname)
        for videoname in os.listdir(idpath):
            videopath = os.path.join(idpath,videoname)
            frame_names = [os.path.join(videopath,f) for f in os.listdir(videopath) if f.endswith('.mp4')]
            info_names = [os.path.join(videopath,f) for f in os.listdir(videopath) if f.endswith('.npy')]
            if len(frame_names) == 0:
                continue
            q.put([frame_names[0],info_names,save_path,videoname])


def process_frame(q1):
    face_detector = FaceDetector(device='cuda')
    output_size = [512,512]
    kk = 1
    def detect_faces(images):
        images = np.stack(images).transpose(0,3,1,2).astype(np.float32)
        images_torch = torch.tensor(images)
        return face_detector.detect_from_batch(images_torch.cuda())
    while True:
        frame_path,info_names,save_base,videoname = q1.get()
        if frame_path is None:
            break 
        video_reader = imageio.get_reader(frame_path)
        for k,info_name in enumerate(info_names):
            info = np.load(info_name)
            save_path = os.path.join(save_base,'%s-%04d'%(videoname,k))
            os.makedirs(save_path,exist_ok=True)
            for (i,x,y,w,h) in info:
                try:
                    img = video_reader.get_data(int(i))
                    height,width,_ = img.shape
                    x,y,w,h = list(map(lambda x:float(x),[x,y,w,h]))
                    i = int(i)
                    box = [x*width,y*height,(x+w)*width,(y+h)*height]
                    if os.path.exists(os.path.join(save_path,'%04d.png'%i)):
                        continue
                    
                    bboxes = detect_faces([img])[0]
                    
                    bbox = choose_one_detection(bboxes,box)
                    if bbox is None:
                        continue 
                    bbox = bbox[:4]
                
                    # Make bbox square and scale it
                    l, t, r, b = bbox
                    SCALE = 1.8

                    center_x, center_y = (l + r) * 0.5, (t + b) * 0.5
                    height, width = b - t, r - l
                    new_box_size = max(height, width)
                    l = center_x - new_box_size / 2 * SCALE
                    r = center_x + new_box_size / 2 * SCALE
                    t = center_y - new_box_size / 2 * SCALE
                    b = center_y + new_box_size / 2 * SCALE

                    # Make floats integers
                    l, t = map(math.floor, (l, t))
                    r, b = map(math.ceil, (r, b))

                    # After rounding, make *exactly* square again
                    b += (r - l) - (b - t)
                    assert b - t == r - l

                    # Make `r` and `b` C-style (=exclusive) indices
                    r += 1
                    b += 1

                    # Crop
                    image_cropped = crop_with_padding(img, t, l, b, r)
                    # Resize to the target resolution
                    image_cropped = cv2.resize(image_cropped, output_size,
                        interpolation=cv2.INTER_CUBIC if output_size[1] > bbox[3] - bbox[1] else cv2.INTER_AREA)

                    # if i == 1319:
                    #     img = cv2.rectangle(img,[int(box[0]),int(box[1])],[int(box[2]),int(box[3])],[0,255,0],10)
                    #     image_cropped = cv2.rectangle(img,[int(bbox[0]),int(bbox[1])],[int(bbox[2]),int(bbox[3])],[255,0,0],10)
                    cv2.imwrite(os.path.join(save_path,'%04d.png'%i),image_cropped[...,::-1])
                    print('\r have done %06d'%i,end='',flush=True)
                    kk += 1
                except:
                    continue
    
        video_reader.close()
    print()

def process_img(q):
    kk = 1
    face_detector = FaceDetector(device='cuda')
    output_size = [512,512]
    def detect_faces(images):
        images = np.stack(images).transpose(0,3,1,2).astype(np.float32)
        images_torch = torch.tensor(images)
        return face_detector.detect_from_batch(images_torch.cuda())
    
    while True:
        img,save_path,i,box = q.get()
        print('*****',save_path,i)
        if img is None:
            break 
        if os.path.exists(os.path.join(save_path,'%04d.png'%i)):
            continue
        
        bboxes = detect_faces([img])[0]
        
        bbox = choose_one_detection(bboxes,box)
        if bbox is None:
            continue 
        bbox = bbox[:4]
       
        # Make bbox square and scale it
        l, t, r, b = bbox
        SCALE = 1.8

        center_x, center_y = (l + r) * 0.5, (t + b) * 0.5
        height, width = b - t, r - l
        new_box_size = max(height, width)
        l = center_x - new_box_size / 2 * SCALE
        r = center_x + new_box_size / 2 * SCALE
        t = center_y - new_box_size / 2 * SCALE
        b = center_y + new_box_size / 2 * SCALE

        # Make floats integers
        l, t = map(math.floor, (l, t))
        r, b = map(math.ceil, (r, b))

        # After rounding, make *exactly* square again
        b += (r - l) - (b - t)
        assert b - t == r - l

        # Make `r` and `b` C-style (=exclusive) indices
        r += 1
        b += 1

        # Crop
        image_cropped = crop_with_padding(img, t, l, b, r)
        # Resize to the target resolution
        image_cropped = cv2.resize(image_cropped, output_size,
            interpolation=cv2.INTER_CUBIC if output_size[1] > bbox[3] - bbox[1] else cv2.INTER_AREA)

        # if i == 1319:
        #     img = cv2.rectangle(img,[int(box[0]),int(box[1])],[int(box[2]),int(box[3])],[0,255,0],10)
        #     image_cropped = cv2.rectangle(img,[int(bbox[0]),int(bbox[1])],[int(bbox[2]),int(bbox[3])],[255,0,0],10)
        cv2.imwrite(os.path.join(save_path,'%04d.png'%i),image_cropped[...,::-1])
        print('\r have done %06d'%i,end='',flush=True)
        kk += 1
    print()

def choose_one_detection(frame_faces,box):
    """
        frame_faces
            list of lists of length 5
            several face detections from one image

        return:
            list of 5 floats
            one of the input detections: `(l, t, r, b, confidence)`
    """
    frame_faces = list(filter(lambda x:x[-1]>0.9,frame_faces))
    if len(frame_faces) == 0:
        return None
    
    else:
        # sort by area, find the largest box
        largest_area, largest_idx = -1, -1
        for idx, face in enumerate(frame_faces):
            area = compute_iou(box,face)
            # area = abs(face[2]-face[0]) * abs(face[1]-face[3])
            if area > largest_area:
                largest_area = area
                largest_idx = idx
        
        if largest_area < 0.1:
            return None
        
        retval = frame_faces[largest_idx]
        
       
    return np.array(retval).tolist()


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (x0, y0, x1, y1), which reflects
            (top, left, bottom, right)
    :param rec2: (x0, y0, x1, y1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0
        # return intersect / S_rec2

def crop_with_padding(image, t, l, b, r):
    """
        image:
            numpy, np.uint8, (H x W x 3) or (H x W)
        t, l, b, r:
            int

        return:
            numpy, (b-t) x (r-l) x 3
    """
    t_clamp, b_clamp = max(0, t), min(b, image.shape[0])
    l_clamp, r_clamp = max(0, l), min(r, image.shape[1])
    image = image[t_clamp:b_clamp, l_clamp:r_clamp]

    # If the bounding box went outside of the image, restore those areas by padding
    padding = [t_clamp - t, b - b_clamp, l_clamp - l, r - r_clamp]
    if sum(padding) == 0: # = if the bbox fully fit into image
        return image

    image = cv2.copyMakeBorder(image, *padding, cv2.BORDER_REFLECT101)
    assert image.shape[:2] == (b - t, r - l)

    # We will blur those padded areas
    h, w = image.shape[:2]
    y, x = map(np.float32, np.ogrid[:h, :w]) # meshgrids
    
    mask_l = np.full_like(x, np.inf) if padding[2] == 0 else (x / padding[2])
    mask_t = np.full_like(y, np.inf) if padding[0] == 0 else (y / padding[0])
    mask_r = np.full_like(x, np.inf) if padding[3] == 0 else ((w-1-x) / padding[3])
    mask_b = np.full_like(y, np.inf) if padding[1] == 0 else ((h-1-y) / padding[1])

    # The farther from the original image border, the more blur will be applied
    mask = np.maximum(
        1.0 - np.minimum(mask_l, mask_r),
        1.0 - np.minimum(mask_t, mask_b))
    
    # Do blur
    sigma = h * 0.016
    kernel_size = 0
    image_blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # Now we'd like to do alpha blending math, so convert to float32
    def to_float32(x):
        x = x.astype(np.float32)
        x /= 255.0
        return x
    image = to_float32(image)
    image_blurred = to_float32(image_blurred)

    # Support 2-dimensional images (e.g. segmentation maps)
    if image.ndim < 3:
        image.shape += (1,)
        image_blurred.shape += (1,)
    mask.shape += (1,)

    # Replace padded areas with their blurred versions, and apply
    # some quickly fading blur to the inner part of the image
    image += (image_blurred - image) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)

   
    fade_color = np.median(image, axis=(0,1))
    image += (fade_color - image) * np.clip(mask, 0.0, 1.0) 
    
    # Convert back to uint8 for interface consistency
    image *= 255.0
    image.round(out=image)
    image.clip(0, 255, out=image)
    image = image.astype(np.uint8)

    return image


if __name__ == "__main__":

    mp.set_start_method('spawn')
    m = mp.Manager()
    q1 = m.Queue()
    q2 = m.Queue()
   
    base = '../dataset/voceleb2'
    save_base = '../dataset/process'
    process_num = 4

    info_p = Process(target=get_video_info,args=(base,save_base,q1,))
    # split_p = Process(target=split_frame,args=(q1,q2,))

    process_list = []
    for _ in range(process_num):
        process_list.append(Process(target=process_frame,args=(q1,)))
     

    info_p.start()
    # split_p.start()
    for p in process_list:
        p.start()

    info_p.join()
    # q1.put([None,None,None,None])
    # split_p.join()
    for _ in range(process_num*2):
         q1.put([None,None,None,None])
    for p in process_list:
        p.join()
    # get_video_info(base,save_base,q1)
    # q1.put([None,None,None,None])
    # split_frame(q1,q2)
    # q2.put([None,None,None,None])
    # process_img(q2)