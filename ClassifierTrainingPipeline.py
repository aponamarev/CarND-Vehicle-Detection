#!/usr/bin/env python
"""
ClassifierTrainingPipeline.py is a pipeline for detecting vehicles in a video
Created 6/10/17.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"

from matplotlib import pyplot as plt
from src.utils import find_cars, add_heat, apply_threshold, draw_labeled_bboxes, draw_bboxes
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import numpy as np
import cv2



class VehicleDetection(object):
    """
    VehicleDetection class provides a pipeline for detecting vehicles over multiple video _frames and visualizing result.
    """
    def __init__(self, classifier, scaller, config, scales=[1.0, 1.5, 2.0, 2.5, 3.0], ystart=400, frame_buffer=4, threshold=6):
        required_config_fields = ['orientations', 'pix_per_cell',
                                  'cell_per_block', 'cmap',
                                  'hist_bins', 'spatial_size']
        for field in required_config_fields:
            assert field in config.keys(), "Error! Provided config does not contain {} filed".format(field)

        self.orien = config['orientations']
        self.pix_per_cell = config['pix_per_cell']
        self.cell_per_block = config['cell_per_block']
        self.cmap = config['cmap']
        self.hist_bins = config['hist_bins']
        self.spatial_size = config['spatial_size']
        self.classifier = classifier
        self.scaller = scaller
        self.scales = scales
        self.ystart = ystart
        self.kernel = 64
        self.kernel2search_factor = 1.25
        self._frames = frame_buffer
        self._bbox_buffer = []
        self._threshold = threshold


    def detect(self, img):
        """
        detect method is designed to detect vehicles in an image in various scales
        :return labeled image, bouding boxes
        """
        # 1. Create a heat map
        heatmap = np.zeros_like(img[:,:,0], dtype=np.uint8)
        # 2. Detect bounding boxes on multiple scales
        multiple_scale_bboxes = []
        for scale in self.scales:
            ystop = int(self.ystart + scale * self.kernel * self.kernel2search_factor)
            bounding_boxes = find_cars(img, self.ystart, ystop, scale, self.classifier, self.scaller,
                                       orient=self.orien,
                                       pix_per_cell=self.pix_per_cell,
                                       cell_per_block=self.cell_per_block,
                                       cmap=self.cmap,
                                       spatial_size=self.spatial_size,
                                       hist_bins=self.hist_bins)
            multiple_scale_bboxes.extend(bounding_boxes)
        # 3. Store bounding boxes into a multi-frame buffer
        self.buffer_bboxes(multiple_scale_bboxes)
        # 4. Map buffer onto the heatmap
        for bboxes in self._bbox_buffer:
            heatmap = add_heat(heatmap, bboxes)

        # 5. Apply threshold
        heatmap = apply_threshold(heatmap, self._threshold)
        # 6. Label resulting heat map
        labels = label(heatmap)
        # 7. Visualize labels on the original image
        if labels[1]!=0:
            img = draw_labeled_bboxes(img, labels)

        return img, labels[0], labels[1]


    def process_video_at(self, path, to_file, debugging=False):

        f = lambda im: self._process_img(im, debugging)

        clip = VideoFileClip(path)
        processed = clip.fl_image(f)
        processed.write_videofile(to_file, audio=False)


    def _process_img(self, img, debugging=False):

        img, heatmap, objects = self.detect(img)

        if debugging:
            if heatmap.max()>0:
                heatmap *= int(255/heatmap.max())
            heatmap = cv2.merge((heatmap, heatmap, heatmap))
            img = np.hstack((img, heatmap))

        return img


    def buffer_bboxes(self, bboxes):
        self._current_bboxes = bboxes
        self._bbox_buffer.append(bboxes)
        if len(self._bbox_buffer)>self._frames:
            self._bbox_buffer.pop(0)



def main():

    import pickle
    # Import pretrained classifier and scaller
    config_path = 'pipeline.p'
    with open(config_path, 'rb') as f:
        pipeline = pickle.load(f)
    scaller = pipeline['scaller']
    classifier = pipeline['classifier']
    feature_config = pipeline['feature_config']

    path = 'test_images/multiscale.png'

    test_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    bounding_boxes = find_cars(test_img, 400, 656, 2.0, classifier, scaller,
                               orient=9, pix_per_cell=8,
                               cell_per_block=2, cmap="YCr",
                               spatial_size=(32,32), hist_bins=32)

    test_img = draw_bboxes(test_img, bounding_boxes)

    plt.imshow(test_img)



    vehicle = VehicleDetection(classifier, scaller, feature_config,
                               frame_buffer=8, threshold=10, scales=[1.5, 2.0, 2.5, 3.0])

    vehicle.process_video_at("project_video.mp4", "test_images/bbox_processed_video.mp4", debugging=False)

    return True

if __name__=="__main__":
    main()
