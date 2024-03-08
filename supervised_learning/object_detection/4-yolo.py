#!/usr/bin/env python3
"""
The building class
for the yolov3 algorithm
allowing it to make predictions and load the data
"""

import tensorflow.keras as K
import numpy as np
import os
import cv2


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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []
        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            threshold = self.class_t
            box_class_scores = np.max(scores, axis=-1)
            positions = np.where(box_class_scores >= threshold)
            filtered_boxes.append(boxes[i][positions])
            box_classes.append(np.argmax(box_class_probs[i],
                                         axis=-1)[positions])
            box_scores.append(box_class_scores[positions])
        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes, axis=-1)
        box_scores = np.concatenate(box_scores, axis=-1)
        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        """
        def calculate_iou(box, boxes):
            """
            Calculate IoU (Intersection over Union) between
            a box and a list of boxes
            """
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            length_x = (boxes[:, 2] - boxes[:, 0])
            length_y = (boxes[:, 3] - boxes[:, 1])
            boxes_area = length_x * length_y
            inter_x1 = np.maximum(box[0], boxes[:, 0])
            inter_y1 = np.maximum(box[1], boxes[:, 1])
            inter_x2 = np.minimum(box[2], boxes[:, 2])
            inter_y2 = np.minimum(box[3], boxes[:, 3])
            inter_x = np.maximum(0, inter_x2 - inter_x1)
            inter_y = np.maximum(0, inter_y2 - inter_y1)
            inter_area = inter_x * inter_y
            iou = inter_area / (box_area + boxes_area - inter_area + 1e-9)
            return iou

        selected_boxes, selected_classes, selected_scores = [], [], []
        for class_idx in np.unique(box_classes):
            class_indices = np.where(box_classes == class_idx)[0]
            class_boxes = filtered_boxes[class_indices]
            class_scores = box_scores[class_indices]
            sorted_indices = np.argsort(class_scores)[::-1]
            class_boxes = class_boxes[sorted_indices]
            class_scores = class_scores[sorted_indices]
            while len(class_boxes) > 0:
                selected_boxes.append(class_boxes[0])
                selected_classes.append(class_idx)
                selected_scores.append(class_scores[0])
                iou = calculate_iou(class_boxes[0], class_boxes[1:])
                filtered_indices = np.where(iou < self.nms_t)[0]
                class_boxes = class_boxes[1:][filtered_indices]
                class_scores = class_scores[1:][filtered_indices]
        selected_boxes = np.array(selected_boxes)
        selected_classes = np.array(selected_classes)
        selected_scores = np.array(selected_scores)
        return selected_boxes, selected_classes, selected_scores

    def load_images(folder_path):
        """
        """
        images_paths = os.listdir(folder_path)
        images = ()
        images_path_tuple = ()
        for image_path in images_paths:
            image_path = os.path.join(folder_path, image_path)
            image = cv2.imread(image_path)
            images = images + image
            images_path_tuple = images_path_tuple + image_path
        return images, images_path_tuple
