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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from numba import jit, cuda #This import is only needed if running on local NOTE: may not work on some builtins
from timeit import default_timer as timer # used for timing model runtime


def process_image(image_path):
    ''' processes the image into tensor

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
    # Save the tensor to a .npy file
    np.save(r'C:\Users/caeden\gitrepos\meltpool_segment_and_chords\DINOV2_Model\saved_Image_process.npy', total_features.numpy())

    print("IMAGE PROCESS TIME : ", timer() - start)
    print(total_features.size())
    return total_features

def pca_formatting(total_features, batch_count):
    ''' process total features into a Principal component analysis of features
    which can be used later on for image representation.

    :param total_features: a tensor rep of all features
    :return:  the formatted PCA features
    '''
    total_features = total_features.reshape(batch_count * patch_h * patch_w,feat_dim)  # count(*H*w, 1024)

    pca = PCA(n_components=3)
    pca.fit(total_features)
    pca_features = pca.transform(total_features)
    return pca_features,pca


def histogram_generation(pca_features):
    ''' visualize PCA components into histogram
        each histogram represents a component
    :param pca_features: the formatted PCA features 3 components
    :display: displays a histogram for each component
    '''

    plt.subplot(2, 2, 1)
    plt.hist(pca_features[:, 0])
    plt.subplot(2, 2, 2)
    plt.hist(pca_features[:, 1])
    plt.subplot(2, 2, 3)
    plt.hist(pca_features[:, 2])
    plt.show()
    plt.close()

def SS_pca_visual(pca_features,batch_count):
    ''' displays single step pca visualization based on the 3 components

    :param pca_features:the formatted PCA features 3 components
    :display: single step pca visualization
    '''
    # min_max scale
    pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / \
(pca_features[:, 0].max() - pca_features[:, 0].min())
    # pca_features = sklearn.processing.minmax_scale(pca_features)
    row_column_count = batch_count//2
    for i in range(batch_count):
        plt.subplot(row_column_count, row_column_count, i + 1)
        plt.imshow(
            pca_features[i * patch_h * patch_w: (i + 1) * patch_h * patch_w,
            0].reshape(patch_h, patch_w))

    plt.show()

def image_seperation(pca_features,pca,batch_count):

    pca_features_bg = pca_features[:, 0] > 0.35  # .35 is threshold
    pca_features_fg = ~pca_features_bg # inverse of prior

    row_column_count = batch_count // 2
    # plot the pca_features_bg
    for i in range(batch_count):
        plt.subplot(row_column_count, row_column_count, i + 1)
        plt.imshow(pca_features_bg[
                   i * patch_h * patch_w: (i + 1) * patch_h * patch_w].reshape(
            patch_h, patch_w))
    plt.show()

    print(len(pca_features_fg), total_features.shape[0])
    print(total_features.shape)

    # 2nd PCA for only foreground patches
    pca_features_fg = pca_features_fg.flatten()
    pca.fit(total_features[pca_features_fg])
    pca.fit(total_features[pca_features_fg])
    pca_features_left = pca.transform(total_features[pca_features_fg])

    for i in range(3):
        # min_max scaling
        pca_features_left[:, i] = (pca_features_left[:, i] - pca_features_left[
                                                             :, i].min()) / (
                                              pca_features_left[:,
                                              i].max() - pca_features_left[:,
                                                         i].min())

    pca_features_rgb = pca_features.copy()
    # for black background
    pca_features_rgb[pca_features_bg] = 0
    # new scaled foreground features
    pca_features_rgb[pca_features_fg] = pca_features_left

    # reshaping to numpy image format
    pca_features_rgb = pca_features_rgb.reshape(batch_count, patch_h, patch_w, 3)
    for i in range(batch_count):
        plt.subplot(row_column_count, row_column_count, i + 1)
        plt.imshow(pca_features_rgb[i])

    plt.show()



def file_counter(folder_path):
    '''
        takes in a folder and returns number of standard image files held
    :param folder_path: self explanatory ( where the folder is )
    :return: integer n images inside of folder
    '''

    try:
        # Get a list of all files in the folder
        all_files = os.listdir(folder_path)

        # Filter for image files (assuming common image extensions like jpg, png, etc.)
        image_files = [file for file in all_files if file.lower().endswith(
            ('.jpg', '.jpeg', '.png', '.gif', '.bmp','.tif'))]

        # Count the number of image files
        number_of_items = len(image_files)

        print(f'The folder contains {number_of_items} image files.')
        return number_of_items

    except FileNotFoundError:
        print(f"Folder not found at location: {folder_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

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

    folder_path = input("Enter the path of the folder containing image files enter N to access saved_Image_process: ")
    # entering exact location and not folder may bring error


    batch_count = file_counter(folder_path)
    if folder_path.lower() != "n":
        total_features = process_image(folder_path)
        with open("saved_batch_count.txt", "w") as file:
            file.write(str(batch_count))
    elif folder_path.lower() == "n":
        # Load the tensor from the .npy file
        loaded_array = np.load(r'C:\Users\caeden\gitrepos\meltpool_segment_and_chords\DINOV2_Model\saved_Image_process.npy')
        # Convert the NumPy array back to a PyTorch tensor
        total_features = torch.from_numpy(loaded_array)
        with open("saved_batch_count.txt", "r") as file:
            batch_count = int(file.readline().strip())

    pca_features,pca = pca_formatting(total_features, batch_count)
    # the prompt below is mainly used for testing
    image_displays = input("Enter H for historgram generation, "
                        "SSP for single step pca, or B for both, other to skip: ")

    if image_displays.lower() == 'h':
        histogram_generation(pca_features)
    elif image_displays.lower() == 'ssp':
        SS_pca_visual(pca_features,batch_count)
    elif image_displays.lower() == 'b':
        histogram_generation(pca_features)
        SS_pca_visual(pca_features,batch_count)

    image_seperation(pca_features,pca,batch_count)
#C:\Users\caeden\gitrepos\meltpool_segment_and_chords\Testing