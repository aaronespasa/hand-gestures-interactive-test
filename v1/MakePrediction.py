import requests
import argparse
import numpy as np
from numpy import floor
import random
import cv2

# Tensorflow
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential,Model
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.losses import categorical_crossentropy
from keras.applications.inception_v3 import InceptionV3
tf.random.set_seed(2)

class MakePrediction:
    def __init__(self):
        # Dictionary
        self.labels = list('abcdefghijklmnopqrstuvwxyz ')
        self.labels.append('delete')
        self.labels.append('nothing')

        # Variables for the model
        self.target_size = (224, 224)
        self.target_dims = (224, 224, 3)  # add channel for RGB
        self.n_classes = 29
        self.val_frac = 0.1
        self.batch_size = 64

        # Variables for OpenCV
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (10, 50)
        self.fontScale = 2
        self.fontColor = (0, 0, 0)
        self.lineType = 2

        # Making of the model
        self.base_model = InceptionV3(include_top=False, weights=None, input_shape=self.target_dims, pooling='max')
        self.dr = Dropout(0.5)(self.base_model.output)
        self.d1 = Dense(512, activation='relu', kernel_regularizer=regularizers.l1(0.001))(self.dr)
        self.d2 = Dense(self.n_classes, activation='softmax')(self.d1)
        self.vggmodel = Model(self.base_model.input, self.d2)

        self.vggmodel.load_weights('./models/vggmodel.h5')
        self.vggmodel.compile(optimizer='adam', loss=categorical_crossentropy, metrics=["accuracy"])

    def load_image(self, imframe):
        img = np.asarray(cv2.resize(imframe, self.target_size))
        mean, std = img.mean(), img.std()
        img = (img - mean) / std
        return img

    def predict(self, frame):
        yhat = np.argmax(self.vggmodel.predict(np.asarray([self.load_image(frame)])))
        return self.labels[yhat]


        #if self.labels[yhat] in ['z', 'l', 'i']:
        #    return 1
        #elif self.labels[yhat] in ['u', 'v']:
        #    return 2
        #elif self.labels[yhat] == 'w':
        #    return 3
        #else
        #    return None
