# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains the files for the Behavioral Cloning Project.

In this project, I used what I've learned about deep neural networks and convolutional neural networks to clone driving behavior. 
I trained, validated and tested a model using Keras. The model outputs a steering angle to an autonomous vehicle.

I used image data and steering angles obtained from [Udacity's provided simulator](https://github.com/udacity/self-driving-car-sim) to train a neural network and then use this model to drive the car autonomously around the simulator's track.

I also created a detailed writeup of the project.

The project's core consists of the following five files:
* [model.py](model.py) (script used to create and train the model)
* [drive.py](drive.py) (script to drive the car - feel free to modify this file)
* `model.h5` (a trained Keras model)
* a report [writeup](writeup_report.md) file (either markdown or pdf)
* `video.mp4` (a video recording of your vehicle driving autonomously around the track for at least one full lap)

The Project
---
The achieved goals / steps of this project are the following:
* Collection of data of good driving behavior from [Udacity's provided simulator](https://github.com/udacity/self-driving-car-sim)
* Designing, training and validating a model that predicts a steering angle from image data
* Using the model to drive the vehicle autonomously around the first track in the simulator. The vehicle remains on the road for (at least) an entire loop around the track.
* Summarizing the results with a written [report](writeup_report.md)

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)
The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` required the existence of the trained model as an h5 file `model.h5`.
See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.
