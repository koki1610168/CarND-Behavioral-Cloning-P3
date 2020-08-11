import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import csv 
import numpy as np
import os

from numpy import savetxt
from numpy import loadtxt
DATADIR = './data/data/'
CSVFILE = DATADIR + 'driving_log.csv'
BATCH = 32

lines = []
with open(CSVFILE) as input:
    reader = csv.reader(input)
    for line in reader:
        lines.append(line)
lines = lines[1:]

import sklearn
from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, BATCH):
    num_samples = len(samples)

    while 1:
        sklearn.utils.shuffle(samples)

        for offset in range(0, num_samples, BATCH):
            batch_samples = samples[offset:offset+BATCH]
            images = []
            angles = []

            for batch_sample in batch_samples:
                filename_center = batch_sample[0]
                filename_left = batch_sample[1]
                filename_right = batch_sample[2]

                path_center = DATADIR + filename_center.strip()
                path_left = DATADIR + filename_left.strip()
                path_right = DATADIR + filename_right.strip()
                
                image_center = mpimg.imread(path_center)
                image_left = mpimg.imread(path_left)
                image_right = mpimg.imread(path_right)

                image_flipped = np.copy(np.fliplr(image_center))

                images.append(image_center)
                images.append(image_left)
                images.append(image_right)
                images.append(image_flipped)

                correction = 0.065
                angle_center = float(batch_sample[3])
                angle_left= angle_center + correction
                angle_right = angle_center - correction
                angle_flipped = -angle_center

                angles.append(angle_center)
                angles.append(angle_left)
                angles.append(angle_right)
                angles.append(angle_flipped)
        
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, BATCH)
validation_generator = generator(validation_samples, BATCH)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout,  Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x/255. - 0.5))
model.add(Convolution2D(filters=24, kernel_size=5, strides=2, activation='relu'))
model.add(Convolution2D(filters=36, kernel_size=5, strides=2, activation='relu'))
model.add(Convolution2D(filters=48, kernel_size=5, strides=2, activation='relu'))
model.add(Convolution2D(filters=64, kernel_size=3, strides=1, activation='relu'))
model.add(Convolution2D(filters=64, kernel_size=3, strides=1, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

train_steps = np.ceil(len(train_samples)/32).astype(np.int32)
validation_steps = np.ceil(len(validation_samples)/32).astype(np.int32)

model.fit_generator(train_generator, steps_per_epoch=train_steps, epochs=5, verbose=1,
        callbacks=None, validation_data=validation_generator, 
        validation_steps=validation_steps, class_weight=None, 
        )


model.save('model1.h5')

model.summary()