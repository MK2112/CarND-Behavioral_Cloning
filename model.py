import os
import csv
import cv2
import gdown     # 'pip install gdown' for Google Drive connectivity
import sklearn
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D

# Read a .csv file's rows, the file and its destination being specified by file_path
# Used as part of process_csv(file_path, folder_path)
def read_csv(file_path):
    lines = []
    with open(file_path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)
    return lines

# Take the path to the .csv file and the path to the image containing folder, read
# .csv file's contents and update paths with folder_path (as .csv file carries wrong paths)
# Returns sets of center, left, right image paths and respective steering angles
def process_csv(file_path, folder_path):
    # Read the .csv file's rows
    lines = read_csv(file_path)
    # Steering angle information needs to get processed differently
    # depending on whether image is center, left or right
    # Paths for center images
    center = []
    # Paths for left side images
    left = []
    # Paths for right side images
    right = []
    # Steering angles per set of simultanuously taken left, right, center images
    steering = []
    # .csv file carries wrong file paths, we update them here
    for line in lines:
        center.append(folder_path + line[0].split('\\')[-1])
        left.append(folder_path + line[1].split('\\')[-1])
        right.append(folder_path + line[2].split('\\')[-1])
        steering.append(float(line[3]))
    return (center, left, right, steering)

# Combine center, left, and right image path sets,
# Correcting left and right image associated steering angle
def correct_steering(center, left, right, steering_in, correction):
    img_paths = []
    img_paths.extend(center)
    img_paths.extend(left)
    img_paths.extend(right)
    steering = []
    steering.extend(steering_in)
    steering.extend([angle + correction for angle in steering])
    steering.extend([angle - correction for angle in steering])
    return (img_paths, steering)

# Generator pulls in only pieces of data, one after another during training,
# this increases memory efficiency
def generator(lines, batch_size=32):
    num_lines = len(lines)
    while 1: # Generator never terminates
        lines = sklearn.utils.shuffle(lines)
        for offset in range(0, num_lines, batch_size):
            batch_lines = lines[offset:offset + batch_size]
            imgs = []
            angles = []
            for img_path, steering in batch_lines:
                # Only here do we read the image with the given image path from 'lines'
                # which is the provided path set (training or validation set)
                # Directly convert read-in BGR image to RGB
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                # Only here do we build a set of actual images and related steering angle labels
                imgs.append(img)
                angles.append(steering)
                # Add vertically flipped versions aswell
                imgs.append(cv2.flip(img,1))
                angles.append(steering * (-1.0))
            # Convert to Numpy arrays for Keras-compatibility
            inp = np.array(imgs)
            out = np.array(angles)
            yield sklearn.utils.shuffle(inp, out)

# Heavily inspired by the Nvidia Dave-2 network achitecture
# as was also suggested by Paul Heraty's forum post.
def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

##
## From here on out we will utilize the above defined functions.
##

# Get dataset if none could be found on the system already
if (not os.path.isdir('/opt/Track/')):
    # Download the dataset
    url = 'https://drive.google.com/uc?id=1NjSAoZ5eI9FX7bwmnvEwoSQHObLb-5qN'
    zip_path = '/opt/Track.zip'
    print('<< Downloading data set')
    gdown.download(url, zip_path, quiet=False)
    print('<< Download successful')
    # Unzip data set
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('/opt/')
else:
    print('<< Data set was found')
    
# Define paths to .csv file and image set
csv_path = '/opt/Track/driving_log.csv'
img_path = '/opt/Track/IMG/'

# Process contents of data set's csv file, add angle to left and right images
center_paths, left_paths, right_paths, steering = process_csv(csv_path, img_path)

# Correct left and right image associated steering angles by an equal factor
img_paths, steering = correct_steering(center_paths, left_paths, right_paths, steering, 0.2)

# Show the data set's size
print('Total Images:', len(img_paths))

# Split up the overall set of image paths and associated steering angles for training and validation
set_list = list(zip(img_paths, steering))
training_set, validation_set = train_test_split(set_list, test_size=0.2)

# Show respective set sizes
print('Training set size:', len(training_set))
print('Validation set size:', len(validation_set))

# Build the generators for training- and validation set
training_generator = generator(training_set, batch_size=32)
validation_generator = generator(validation_set, batch_size=32)

# Setup the model
model = get_model()

# Compiling and training the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(training_generator, steps_per_epoch=len(training_set), validation_steps=len(validation_set), validation_data=validation_generator, epochs=3, verbose=1)

# Save the model
model.save('model.h5')

# Show additional data gathered from training (loss on training set, loss on validation set)
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])