#!/usr/bin/env python3
"""
The building class
for the yolov3 algorithm
allowing it to make predictions and load the data
"""

import tensorflow.keras as K

class Yolo:
    model = None
    class_names = None
    class_t = None
    nms_t = None
    anchors = None
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        with open(classes_path, 'r') as file:
            classes_list = [line.strip() for line in file if line.strip()]
        self.model = K.models.load_model(model_path)
        self.class_names = classes_list
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
