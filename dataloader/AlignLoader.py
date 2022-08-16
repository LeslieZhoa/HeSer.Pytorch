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

        # source root
        root = kwargs['root']
        self.paths = [os.path.join(root,f) for f in os.listdir(root)]
        dis = math.floor(len(self.paths)/self.count)
        self.paths = self.paths[self.id*dis:(self.id+1)*dis]
        self.length = len(self.paths)
        random.shuffle(self.paths)
        self.frame_num = kwargs['frame_num']

       

    def __getitem__(self,i):
        
        idx = i % self.length
        id_path = self.paths[idx]
        video_paths = [os.path.join(id_path,f) for f in os.listdir(id_path)]
        vIdx = random.randint(0, len(video_paths) - 1)
        video_path = video_paths[vIdx]
        img_paths = [os.path.join(video_path,f) for f in os.listdir(video_path)]
        iidx = sorted(random.sample(range(len(img_paths)),self.frame_num))

        t_img_path = img_paths[iidx[-1]]

        xs = []
        for i in iidx[:-1]:
            with Image.open(img_paths[i]) as img:
                xs.append(self.norm(self.transform(img.convert('RGB'))).unsqueeze(0))
        xs = torch.cat(xs,0)
        with Image.open(t_img_path) as img:
            xt = self.transform(img.convert('RGB'))
        
        xt,gt = self.aug_fn.augment_double(xt,xt)
        xt = self.norm(xt)
        gt = self.norm(gt)
       
        return xs,xt,gt


    def __len__(self):
        # return self.length
        return 4

