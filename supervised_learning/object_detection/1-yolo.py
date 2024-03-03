#!/usr/bin/env python3
"""
The building class
for the yolov3 algorithm
allowing it to make predictions and load the data
"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    Yolo class to define the elements
    and functions making the algorithm work
    """
    model = None
    class_names = None
    class_t = None
    nms_t = None
    anchors = None

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        initialize the attributes for the yolo
        algorithm
        """
        with open(classes_path, 'r') as file:
            classes_list = [line.strip() for line in file if line.strip()]
        self.model = K.models.load_model(model_path)
        self.class_names = classes_list
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
    def process_outputs(self, outputs, image_size):
        """
        Using the method cited in the research paper:
        https://pjreddie.com/media/files/papers/YOLOv3.pdf
        to get the bounding boxes out of the output
        of the yolo model
        """
        def sigmoid(x):
            z = 1/(1 + np.exp(-x))
            return z
        boxes, box_confidences, box_class_probs = [], [], []
        for index, output in enumerate(outputs):
            grid_h = output.shape[0]
            grid_w = output.shape[1]
            box_confidences.append(sigmoid(output[..., 4:5]))
            box_class_probs.append(sigmoid(output[:, :, :, 5:]))
            box = np.zeros(output[:, :, :, :4].shape)
            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]

            pw_total = self.anchors[:, :, 0]
            pw = np.tile(pw_total[index], grid_w)
            pw = pw.reshape(grid_w, 1, len(pw_total[index]))
            ph_total = self.anchors[:, :, 1]
            ph = np.tile(ph_total[index], grid_h)
            ph = ph.reshape(grid_h, 1, len(ph_total[index]))
            cx = np.tile(np.arange(grid_w), grid_h)
            cx = cx.reshape(grid_w, grid_w, 1)
            cy = np.tile(np.arange(grid_w), grid_h)
            cy = cy.reshape(grid_h, grid_h).T
            cy = cy.reshape(grid_h, grid_h, 1)

            bx = (1 / (1 + np.exp(-t_x))) + cx
            by = (1 / (1 + np.exp(-t_y))) + cy
            bw = np.exp(t_w) * pw
            bh = np.exp(t_h) * ph
            bx = bx / grid_w
            by = by / grid_h
            bw = bw / self.model.input.shape[1]
            bh = bh / self.model.input.shape[2]

            x1 = (bx - (bw / 2)) * image_size[1]
            y1 = (by - (bh / 2)) * image_size[0]
            x2 = (bx + (bw / 2)) * image_size[1]
            y2 = (by + (bh / 2)) * image_size[0]
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2
            boxes.append(box)
        return boxes, box_confidences, box_class_probs