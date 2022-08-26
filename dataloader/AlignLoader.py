#! /usr/bin/python 
# -*- encoding: utf-8 -*-
'''
@author zhaoxiang
@date 20220812
'''

import os 

from torchvision import transforms 
import PIL.Image as Image
from dataloader.DataLoader import DatasetBase
import random
import math
import torch
from dataloader.augmentation import ParametricAugmenter
import numpy as np


class AlignData(DatasetBase):
    def __init__(self, slice_id=0, slice_count=1,dist=False, **kwargs):
        super().__init__(slice_id, slice_count,dist, **kwargs)


        self.transform = transforms.Compose([
            transforms.Resize((kwargs['size'], kwargs['size'])),
             transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        self.aug_fn = ParametricAugmenter(use_pixelwise_augs=kwargs['use_pixelwise_augs'],
                                        use_affine_scale=kwargs['use_affine_scale'],
                                        use_affine_shift=kwargs['use_affine_shift'])

        
        self.norm = transforms.Compose([transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

        self.resize = transforms.Compose([
            transforms.Resize((256,256))])

        # source root
        root = kwargs['root']
        self.paths = [os.path.join(root,f) for f in os.listdir(root)]
        # dis = math.floor(len(self.paths)/self.count)
        # self.paths = self.paths[self.id*dis:(self.id+1)*dis]
        self.length = len(self.paths)
        random.shuffle(self.paths)
        self.frame_num = kwargs['frame_num']
        self.skip_frame = kwargs['skip_frame']
        self.eval = kwargs['eval']
        self.size = kwargs['size']
        self.scale = 0.4 / 1.8

       

    def __getitem__(self,i):
        
        idx = i % self.length
        id_path = self.paths[idx]
        video_paths = [os.path.join(id_path,f) for f in os.listdir(id_path)]
        vIdx = random.randint(0, len(video_paths) - 1)
        video_path = video_paths[vIdx]
        img_paths = [os.path.join(video_path,f) for f in os.listdir(video_path)]
        begin_idx = random.randint(0, len(img_paths) - self.frame_num*self.skip_frame - 1)
        img_paths = [img_paths[i] 
                    for i in range(begin_idx,begin_idx+self.frame_num*self.skip_frame,self.skip_frame)] 

        s_img_paths = img_paths[:-1]

        t_img_path = img_paths[-1]

        xs = []
        for img_path in s_img_paths:
            with Image.open(img_path) as img:
                xs.append(self.norm(self.transform(img.convert('RGB'))).unsqueeze(0))
        xs = torch.cat(xs,0)

        if self.eval:
            idx = (i + random.randint(0,self.length-1)) % self.length
            id_path = self.paths[idx]
            video_paths = [os.path.join(id_path,f) for f in os.listdir(id_path)]
            vIdx = random.randint(0, len(video_paths) - 1)
            video_path = video_paths[vIdx]
            img_paths = [os.path.join(video_path,f) for f in os.listdir(video_path)]
            t_img_path = img_paths[random.randint(0, len(img_paths) - 1)]

        with Image.open(t_img_path) as img:
            xt = self.transform(img.convert('RGB'))
        
        mask = np.zeros((self.size,self.size,3),dtype=np.uint8)
        mask[int(self.size*self.scale):int(-self.size*self.scale),
            int(self.size*self.scale):int(-self.size*self.scale)] = 255
        mask = mask[np.newaxis,:]
        xt,gt,mask = self.aug_fn.augment_triple(xt,xt,mask)
        indexs = torch.where(mask==1)
        top = indexs[1].min()
        bottom = indexs[1].max()
        left = indexs[2].min()
        right = indexs[2].max()
        crop_xt = xt[...,top:bottom,left:right]
        crop_xt = self.norm(crop_xt)
        xt = self.norm(xt)
        gt = self.norm(gt)
       
        return self.resize(xs),self.resize(xt),self.resize(crop_xt),gt


    def __len__(self):
        if self.eval:
            return 1000
        else:
            # return self.length
            return max(self.length,1000)

