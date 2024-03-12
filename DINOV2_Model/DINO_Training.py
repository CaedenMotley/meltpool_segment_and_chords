"""
File: DINOV2_Processing.py
Author: Caeden Motley
Date: 1/13/24
Description: Pushing images through dinov2 for features (patch_embed)
"""

import os
import sys

sys.path.append(r'C:\Users\caeden\gitrepos\dinov2') # THIS WILL NEED TO BE CHANGED FOR NON LOCAL
import torch
import torchvision
import PIL
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from numba import jit, cuda #This import is only needed if running on local NOTE: may not work on some builtins
from timeit import default_timer as timer # used for timing model runtime


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def process_image(image_path):
    ''' processes the image into tensor

    :param image_path: desired folder of images ANDOR image
    :return: tensor representation of features
    '''
    start = timer()

    total_features = []
    with torch.no_grad(): # disable gradient calc
        for img_path in os.listdir(image_path):
            if(not (img_path.endswith('npy'))):
                img_path = os.path.join(image_path, img_path)
                img = PIL.Image.open(img_path).convert('RGB')
                img = np.array(img)
                img = np.moveaxis(img, -1, 0)
                img_t = np.repeat(np.repeat(img, repeats=14, axis=1), repeats=14,axis=2) # this upscales the image to negate the patch sizing
                img_t = torch.from_numpy(img_t).float()
                #img_t = transform1(img) this is only needed if not upscaling
                features_dict = dinov2_vitl14.forward_features(img_t.unsqueeze(0)) # add a dummy dimension (1) to beggining for proper formatting
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                dir_size = os.path.splitext(os.path.basename(img_path))[1]
                # embed depth for vitl14 = 1024
                features = features_dict['x_norm_patchtokens']

                '''
                # uncomment below if not using upscaled image!

                # uses inferred features from localized images !!! wow this is much easier than hand drawing everything for data!
                #features = features.reshape(cropped_size//patch_size, cropped_size//patch_size, 1024) 
                f#eatures = np.repeat(np.repeat(features, repeats=14,axis=0), repeats=14, axis=1)
                #features = features.reshape((cropped_size, cropped_size, 1024))
                '''
                
                total_features.append(features)
                np.save(r''+ image_path + '\\' f"{img_name}_features.npy",features.numpy())

    # Note: the total_features below WILL NOT WORK ON MACHINES WITH LOW RAM better to comment out if unkown!
    total_features = torch.cat(total_features, dim=0)# concactenates ALL features extracted creating a consolidated representation of all the features.
    total_features = total_features.view(batch_count,-1, 1024)
    # Save the tensor to a .npy file
    #np.save(r''+ image_path + '\\BatchTotalFeatures .npy', total_features.numpy())

    print("IMAGE PROCESS TIME : ", timer() - start)
    print(total_features.size())
    print(total_features.shape)
    return total_features

def pca_formatting(total_features, batch_count):
    ''' process total features into a Principal component analysis of features
    which can be used later on for image representation.

    :param total_features: a tensor rep of all features
    :return:  the formatted PCA features
    '''
    total_features = total_features.reshape(1 * cropped_size * cropped_size,feat_dim)  # Reshape for PCA analysis: batch_count * patch_h * patch_w x features (feat_dim)
   # note this does not rid any features just  reformats to batch_count * patch_h * patch_w rows and feat_dim columns
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
    row_column_count = batch_count//2
    for i in range(1): # CHANGE THE RANGE TO BATCH COUNT IF USING BATCH > 1
        plt.subplot(row_column_count, row_column_count, i + 1)
        plt.imshow(
            pca_features[i * cropped_size * cropped_size: (i + 1) * cropped_size * cropped_size,
            0].reshape(cropped_size, cropped_size))

    plt.show()

def image_seperation(pca_features,pca,batch_count):

    pca_features_bg = pca_features[:, 0] > 0.35  # .35 is threshold
    pca_features_fg = ~pca_features_bg # inverse of prior

    row_column_count = batch_count // 2
    # plot the pca_features_bg
    for i in range(1): # CHANGE THE RANGE TO BATCH COUNT IF USING BATCH > 1
        plt.subplot(row_column_count, row_column_count, i + 1)
        plt.imshow(pca_features_bg[
                   i * cropped_size * cropped_size: (i + 1) * cropped_size * cropped_size].reshape(
            cropped_size, cropped_size))
    plt.show()

    print(len(pca_features_fg), total_features.shape[0])
    print(total_features.shape)







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
        filtered_files = [file for file in image_files if not file.endswith('.npy')]

        # Count the number of image files
        number_of_items = len(filtered_files)

        print(f'The folder contains {number_of_items} image files.')
        return number_of_items

    except FileNotFoundError:
        print(f"Folder not found at location: {folder_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":

    # initialize the dino model
    dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    size = int(input("Please input standard image size (will be cropped to be divisible by 14): "))
    cropped_size = size - (size % 14) #994 if 1000
    transform1 = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(cropped_size), # adjusts to handle patch size being 14
        # should be multiple of model patch_size
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize(mean=0.5, std=0.2)
    ])

    patch_size = dinov2_vitl14.patch_size  # patchsize=14

    # 1000//14
    patch_h = size // patch_size
    patch_w = size // patch_size

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
        # NOTE: this will be finalized for non local machines later
        loaded_array = np.load(r'C:\Users\caeden\gitrepos\meltpool_segment_and_chords\DINOV2_Model\Training_data\DinoV2Features\1000by1000\Training_2_crop88_features.npy') # change this for what you want to load
        # Convert the NumPy array back to a PyTorch tensor
        total_features = torch.from_numpy(loaded_array)
        #new_tensor = total_features.view(215, 215, 1024)

        print(total_features.shape)

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
