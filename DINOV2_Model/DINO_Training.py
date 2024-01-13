"""
File: DINO_Training.py
Author: Caeden Motley
Date: 1/13/24
Description: Training dinov2 on meltpool images
"""
import os

import torch
import torchvision
import PIL

import matplotlib.pyplot as plt
import numpy

from numba import jit, cuda #This import is only needed if running on local NOTE: may not work on some builtins

from timeit import default_timer as timer # used for timing model runtime

# comment this out if not running on local
def process_image(image_path):
    ''' processes the image into tensor of N x M x K

        N: number of images processed


    :param image_path: desired folder of images ANDOR image
    :return: tensor representation of features
    '''
    start = timer()

    total_features = []
    with torch.no_grad():
        for img_path in os.listdir(image_path):
            img_path = os.path.join(image_path, img_path)
            img = PIL.Image.open(img_path).convert('RGB')
            img_t = transform1(img)

            features_dict = dinov2_vitl14.forward_features(img_t.unsqueeze(0))
            features = features_dict['x_norm_patchtokens']
            total_features.append(features)

    total_features = torch.cat(total_features, dim=0)
    total_features.shape

    print("IMAGE PROCESS TIME : ", timer() - start)
    print(total_features.size())

if __name__ == "__main__":

    # initialize the dino model
    dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')


    transform1 = torchvision.transforms.Compose([
        torchvision.transforms.Resize(520),
        torchvision.transforms.CenterCrop(518),
        # should be multiple of model patch_size
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=0.5, std=0.2)
    ])

    patch_size = dinov2_vitl14.patch_size  # patchsize=14

    # 520//14
    patch_h = 520 // patch_size
    patch_w = 520 // patch_size

    feat_dim = 1024  # vitl14

    file_location = input("Enter the path of the image file (ENTER NO IF IMAGE ALREADY PROCESSED): ")
    if file_location.lower() != "no":
        process_image(file_location)






#C:\Users\caeden\Pictures\DinoTest