"""YOLO v3 output
"""
import numpy as np
import keras.backend as K
from keras.models import load_model

yolo3_model = load_model('yolov3-tiny.h5')
yolo3_model.summary()
