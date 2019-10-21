import os
import numpy as np
import random
import cv2
from torch.utils import data
import torch
#import torchvision


class ImageSaliencyDataSet(data.Dataset):
    def __init__(self, imageroot,
                 list_path, 
                 max_iters=None, input_size=(224, 224), mean=(128, 128, 128), crop=True, rotate=True, ignore_label=256):
        self.imageroot = imageroot
        self.list_path = list_path
        self.input_size = input_size
        self.crop = crop
        self.ignore_label = ignore_label
        self.mean = mean
        self.rotate = rotate
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        
        self.img_ids = [img_id.strip() for img_id in open(list_path)]

        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.files = []
        
        for name in self.img_ids:
            img_file = os.path.join(self.imageroot, "Images/%s.jpg" % name)
            label_file = os.path.join(self.imageroot, "GroundTruth/%s.png" % name)

            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)
    
    def rotate_bound(self, image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
    
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
    
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
    
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
    
        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))
    
    def generate_rotate_label(self, image,  label):
        anglelist = [0,90,180,270,1] #angle=270 equals diagonal flip
        angle = anglelist[np.random.choice(5)]
        if angle == 1:
            image = cv2.flip(image,1,dst=None) #horizontal flip
            label = cv2.flip(label,1,dst=None) #horizontal flip
        else:
            image = self.rotate_bound(image, angle)
            label = self.rotate_bound(label, angle)
#        print image.shape,label.shape
        return image, label
    
    def generate_crop_label(self, image, label):
        f_scale = 0.5 + np.random.randint(5) / 10
        img_h = image.shape[0]
        img_w = image.shape[1]
        img_d = image.shape[2]
    
        crop_shape = (int(image.shape[0]*f_scale), int(image.shape[1]*f_scale))
        oshape_h = img_h
        oshape_w = img_w
        img_pad = np.zeros([oshape_h, oshape_w, img_d], np.uint8)
        
        label_pad = np.zeros([oshape_h, oshape_w, img_d], np.uint8)

        img_pad[:img_h, :img_w, 0:img_d] = image
        label_pad[:img_h, :img_w, 0:img_d] = label
#        label_pad[:img_h, :img_w] = label
      
        nh = np.random.randint(0, oshape_h - crop_shape[0])
        nw = np.random.randint(0, oshape_w - crop_shape[1])

        image_crop = img_pad[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
        image_crop = cv2.resize(image_crop,(img_h,img_w))
        
        label_crop = label_pad[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
        label_crop = cv2.resize(label_crop,(img_h,img_w), interpolation = cv2.INTER_NEAREST)

        return image_crop, label_crop
          

    def __getitem__(self, index):
        datafiles = self.files[index]
        # print len(self.files)
        # print datafiles
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.input_size).astype(float)
        label = cv2.resize(label, self.input_size , interpolation = cv2.INTER_NEAREST)
        image_size = image.shape
#        print(label.shape)
        name = datafiles["name"]
        # print frame_size
        name = datafiles["name"]

        if self.crop:
            image, label = self.generate_crop_label(image, label)
        if self.rotate:
            image, label = self.generate_rotate_label(image, label)
            
        image = np.asarray(image, np.float32)
        image = image - self.mean
        
        image = image.transpose((2, 0, 1)) # for flipping image
        
        image = torch.from_numpy(image)

        label[label > 0] = 1
        label = torch.from_numpy(label)
#        return frames.copy(), label.copy(), np.array(frame_size), name
        return image, label, np.array(image_size), name


class ImageSaliencyTestDataSet(data.Dataset):
    def __init__(self, imageroot,
                 list_path, 
                 max_iters=None, input_size=(224, 224), mean=(128, 128, 128), crop=False, rotate=False, ignore_label=256, validation=True):
        self.imageroot = imageroot
        self.list_path = list_path
        self.input_size = input_size
        self.crop = crop
        self.ignore_label = ignore_label
        self.mean = mean
        self.rotate = rotate
        self.validation = validation
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        
#        self.frames_ids = [i_id.strip() for i_id in open(list_path)]
        self.img_ids = [img_id.strip() for img_id in open(list_path)]

        self.files = []
        for name in self.img_ids:
                img_file = os.path.join(self.imageroot, "Images/%s.jpg" % name)
                label_file = os.path.join(self.imageroot, "GroundTruth/%s.png" % name)
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                })


    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        datafiles = self.files[index]
        # print len(self.files)
        # print datafiles
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        if self.validation:
            image = cv2.resize(image, self.input_size).astype(float)
            label = cv2.resize(label,self.input_size , interpolation = cv2.INTER_NEAREST)
            image_size = image.shape
        else:
            image_size = image.shape
            image = cv2.resize(image, self.input_size).astype(float)
            label = cv2.resize(label,self.input_size , interpolation = cv2.INTER_NEAREST)
        name = datafiles["name"]
        # print size
        # name = os.path.splitext(os.path.basename(datafiles["img"]))[0]
        image = np.asarray(image, np.float32)
        image = image - self.mean
        # print label.shape
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        label[label > 0] = 1
        label = torch.from_numpy(label)
        
        return image, label, np.array(image_size),name
