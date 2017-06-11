#!/usr/bin/env python
"""
ClassifierTrainingPipeline.py is a pipeline for extracting features and training linear SVC classifier
Created 6/10/17.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"

from matplotlib import pyplot as plt
from glob import glob
from tqdm import trange
from sys import getsizeof
from skimage.feature import hog
from collections import namedtuple
import numpy as np
import cv2, os

Point = namedtuple("Point", ['x','y'])

def load_imgs(path, format="*.png"):
    path = os.path.join(path,"**",format)
    fullpaths = glob(path, recursive=True)
    print("Begin image loading process...")
    imgs = []
    for i in trange(len(fullpaths)):
        try:
            im = cv2.cvtColor(cv2.imread(fullpaths[i]), cv2.COLOR_BGR2RGB)
            imgs.append(im)
        except:
            pass
    imgs = np.array(imgs)
    print("\n{} images were loaded. Total size of the dataset is {:.1f}Mb. Dataset shape: {}".format(
        len(imgs), round(getsizeof(imgs)/2**20),1), imgs.shape)
    return imgs

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    """Method provided by Udacity as part of Udacity Self-Driving Nanadegree - Project: Vehicle Detection and Tracking"""
    assert len(img.shape)==2, "Incorrect image shape. Image should be y*x one channel image."
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features

def cvt2hog(dataset, withCongig={"orientations": 9, "pix_per_cell": 8, "cell_per_block": 2}):
    print("Initiating conversion to hog.")
    hog_features = []
    for i in trange(len(dataset)):
        c_features = []
        for c in range(dataset.shape[3]):
            c_features.extend(get_hog_features(dataset[i, :, :, c],
                                               orient=withCongig["orientations"],
                                               pix_per_cell=withCongig["pix_per_cell"],
                                               cell_per_block=withCongig["cell_per_block"]))
        hog_features.append(c_features)
    hog_features=np.array(hog_features)
    print("Conversion is complete. Original dataset:", dataset.shape, "was converted into:", hog_features.shape)
    return hog_features


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def img2features(img, orient, pix_per_cell, cell_per_block,
                 cmap="HLS", spatial_size=(32, 32), hist_bins=32):

    img = cvtColor(img, cmap)/255
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(img[:, :, 0], orient, pix_per_cell, cell_per_block, feature_vec=True)
    hog2 = get_hog_features(img[:, :, 1], orient, pix_per_cell, cell_per_block, feature_vec=True)
    hog3 = get_hog_features(img[:, :, 2], orient, pix_per_cell, cell_per_block, feature_vec=True)
    hog_features = np.hstack((hog1, hog2, hog3))
    # Get color features
    #spatial_features = bin_spatial(img, size=spatial_size)
    hist_features = color_hist(img, nbins=hist_bins)
    #features = np.hstack((spatial_features, hist_features, hog_features))
    features = np.hstack((hist_features, hog_features))
    return features

def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
              cmap="HLS", spatial_size=(32, 32), hist_bins=32, window = 64):
    """
    Method provided by Udacity as part of Udacity Self-Driving Nanadegree - Project: Vehicle Detection and Tracking
    :param img: Input RGB image
    :param ystart: To optimise the compute time, the method limits the area of search for cars by restricting search
    :param ystop: through ystart and ystop
    :param scale: factor for scalling down the image (resize(img, (x/scale, y/scale))
    :param svc: Linear SVM classifier
    :param X_scaler: Scaller for features designed to ensure that all features mean centered with std=1
    :param orient: Number of orientation angles for Histogram of Oriented Gradient (HOG)
    :param pix_per_cell: Number of pixels used to calculate HOG
    :param cell_per_block: Number of cells to normalize the image by (used by HOG)
    :param spatial_size: Default (32, 32)
    :param hist_bins: Default 32
    :return: original with bounding box for vehicle
    """
    boudning_boxes = []
    img = cvtColor(img, cmap).astype(np.float32)/255
    img2search = img[ystart:ystop]
    if scale!=1:
        imshape = img2search.shape
        img2search = cv2.resize(img2search, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    # Define blocks and steps as above
    nxblocks = (img2search.shape[1]//pix_per_cell) - cell_per_block + 1
    nyblocks = (img2search.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(img2search[:,:,0], orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(img2search[:,:,1], orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(img2search[:,:,2], orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(img2search[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            #spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            #test_features = np.hstack((spatial_features, hist_features, hog_features)).reshape((1,-1))
            test_features = np.hstack((hist_features, hog_features)).reshape((1, -1))
            test_features = X_scaler.transform(test_features)
            test_prediction = svc.predict(test_features)

            if test_prediction==1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                upper_left = Point(xbox_left, ytop_draw+ystart)
                bottom_right = Point(xbox_left+win_draw, ytop_draw+win_draw+ystart)
                boudning_boxes.append((upper_left, bottom_right))
    return boudning_boxes


def draw_bboxes(img, bbox_list):
    """
    draw_bboxes designed to draw boxes from a given list
    """
    for bbox in bbox_list:
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

    return img


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def cvtColor(img, color="RGB"):
    # Create a number of available color schemes
    color_labels = ["HLS", "HSV","LAB","YUV","YCr"]
    color_option = [cv2.COLOR_RGB2HLS, cv2.COLOR_RGB2HSV, cv2.COLOR_RGB2LAB, cv2.COLOR_RGB2YUV, cv2.COLOR_RGB2YCrCb]
    assert color in color_labels or color=="RGB", "Error: Incorrect color label"
    # Convert original image into a desired color channel
    if color!="RGB":
        img = cv2.cvtColor(img, color_option[color_labels.index(color)])
    return img


def main():

    import pickle
    # Import pretrained classifier and scaller
    config_path = 'pipeline.p'
    with open(config_path, 'rb') as f:
        pipeline = pickle.load(f)
    scaller = pipeline['scaller']
    classifier = pipeline['classifier']
    feature_config = pipeline['feature_config']

    path = "Data/vehicles/GTI_Far/image0000.png"

    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    features = img2features(img,
                            orient=feature_config['orientations'],
                            pix_per_cell=feature_config['pix_per_cell'],
                            cell_per_block=feature_config['cell_per_block'],
                            cmap=feature_config['cmap'],
                            spatial_size=feature_config['spatial_size'],
                            hist_bins=feature_config['hist_bins'])
    features = scaller.transform(features.reshape((1,-1)))
    prediction = classifier.predict(features)
    print(prediction)
    # Test bounding box detection
    test_img_paths = ["test_images/test01.png", "test_images/test02.png",
                      "test_images/test03.png", "test_images/test04.png",
                      "test_images/test05.png", "test_images/test1.jpg",
                      "test_images/test2.jpg", "test_images/test3.jpg",
                      "test_images/test4.jpg", "test_images/test5.jpg"]
    ystart = 400
    scale = 2.0
    ystop = int(ystart + scale*64*1.25)

    for p in test_img_paths:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        bounding_boxes = find_cars(img, ystart, ystop, scale, classifier, scaller,
                                   orient=feature_config['orientations'],
                                   pix_per_cell=feature_config['pix_per_cell'],
                                   cell_per_block=feature_config['cell_per_block'],
                                   cmap = feature_config['cmap'],
                                   spatial_size=feature_config['spatial_size'],
                                   hist_bins=feature_config['hist_bins'])
        bounding_box_image = draw_bboxes(img, bounding_boxes)
        plt.imshow(bounding_box_image)

    return True

if __name__=="__main__":
    main()
