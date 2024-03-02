import numpy as np
import skimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier
from functools import partial

from skimage.feature import canny, blob_dog, peak_local_max
from skimage.util import invert
from skimage import data
from skimage import color
from skimage.filters import meijering, sato, frangi, hessian
from skimage.morphology import skeletonize, thin, dilation, disk,closing
from skimage.segmentation import random_walker
from skimage.filters.rank import median
from skimage import segmentation, feature, future
import time
import os

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
        image_files = [file for file in all_files]
        filtered_files = [file for file in image_files if file.endswith('.npy')]

        # Count the number of image files
        number_of_items = len(filtered_files)

        print(f'The folder contains {number_of_items} image files.')
        return number_of_items,filtered_files

    except FileNotFoundError:
        print(f"Folder not found at location: {folder_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
def create_dataset_dict(feature_files, label_files):
    """
    Create a dictionary with keys "features" and "labels" containing lists of file paths.

    Parameters:
    - feature_files (list): List of feature file paths.
    - label_files (list): List of label file paths.

    Returns:
    - dataset_dict (dict): Dictionary with keys "features" and "labels" containing lists of file paths.
    """

    # Check if the number of feature files and label files match
    if len(feature_files) != len(label_files):
        raise ValueError("Number of feature files must be equal to the number of label files.")

    # Create a dictionary to store the dataset
    dataset_dict = {'features': feature_files, 'labels': label_files}

    return dataset_dict

if __name__ == "__main__":

    #folder_path = input("Enter the path to where ONLY feature files are contained")
    #folder_path = input("Enter the path to where ONLY LABEL files are contained")

    #batch_count_features,feature_files = file_counter(folder_path)
    #batch_count_masks,masks_files = file_counter(folder_path)

    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                 max_depth=10,
                                 max_samples=0.05)  # reuse this line in seperate training!
    # load features of features into (988,036) after 994 x 994
    # look into sample num and feature num from random forest so s x f (samples by features)
    # features = x labels = y

    train1_features = np.asarray(np.load(r'C:\Users\caeden\gitrepos\meltpool_segment_and_chords\DINOV2_Model\Training_data\DinoV2Features\1000by1000\Training_2_crop88_features.npy'))
    train1_features_ = np.reshape(train1_features,
                                  (-1, train1_features.shape[2]))

    train1_mask  = np.load(r'C:\Users\caeden\gitrepos\meltpool_segment_and_chords\Random_Forest_Model\AlSi10Mg-Training\Training_2_mask.npy')
    train1_mask = train1_mask[:994, :994]
    train1_mask = train1_mask.reshape(-1)
    ind1 = (train1_mask > -1)


    ind1_ = (train1_mask == -1)
    y_train1 = train1_mask[ind1]
    X_train1 = train1_features_[ind1, :]

    test_image_raw = Image.open(
        r'C:\Users\caeden\gitrepos\meltpool_segment_and_chords\DINOV2_Model\Training_data\DinoV2Features\1000by1000\Training_2_crop88_rot90.tif')
    test_image = test_image_raw.convert('L')
    test_image = np.asarray(test_image)
    test_image = test_image[:994, :994]



    clf.fit(X_train1,y_train1)   ### copy this as well to fit the dataset (same as stacking in theory) y_hat = clf.predict(same x) once y_hat is predicted reshape into 2d matrix (original w x h) and show side by side ground truth and y_hat

    # Predicting for our test image
    test_features = np.load(r'C:\Users\caeden\gitrepos\meltpool_segment_and_chords\DINOV2_Model\Training_data\DinoV2Features\1000by1000\Training_2_crop88_rot90_features.npy')
    TEST_image = np.reshape(test_features, (-1, test_features.shape[2]))
    X_test = TEST_image

    # Timing
    st = time.time()
    y_test = clf.predict(X_test)
    end = time.time()
    execution = end - st
    print(execution)
    # print ("Check4")

    # Reshaping back into orginal image shape
    remake_img = np.reshape(y_test, test_image.shape)

    # Unique classifications (0 - boundary, 1 - background, 2 - interior)
    values = np.unique(remake_img)

    # Visualization
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
    ax[0].imshow(test_image, cmap=plt.cm.gray)
    ax[0].set_title('Raw Image')
    ax[1].imshow(remake_img)
    ax[1].set_title('Segmentation of crop45.tif')
    fig.tight_layout()
    # plt.savefig('Results\crop45_seg1.png', bbox_inches='tight', pad_inches=0)
    plt.show()
