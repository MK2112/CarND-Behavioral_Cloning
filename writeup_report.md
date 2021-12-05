# **Behavioral Cloning**

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/right_2021_09_04_21_42_09_631.png "Training set image"
[image2]: ./examples/center_2021_09_04_21_17_07_917.jpg "Corner Correction 1"
[image3]: ./examples/center_2021_09_04_21_17_08_000.jpg "Corner Correction 2"
[image4]: ./examples/center_2021_09_04_21_17_08_083.jpg "Corner Correction 3"
[image5]: ./examples/center_2021_09_04_21_17_08_169.jpg "Corner Correction 3"
[image6]: ./examples/center_2021_09_04_21_41_18_461.jpg "Hill Road"
[image7]: ./examples/flip.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

This project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `writeup_report.md` summarizing the results
* `video.mp4` showing the car doing an entire lap around the course


#### 2. Submission includes functional code
Using the Udacity provided simulator and the provided but untouched `drive.py` file,
the car can be driven autonomously around the track by executing this:

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for downloading the data set from Google Drive,
for training and for saving the convolution neural network. The file shows the pipeline I used for
training and validating the model, and it contains comments to explain how the code works and what the intentions were.

Before running the `model.py` make sure to install `gdown` for connectivity to Google Drive:
```sh
pip install gdown
```
I used `gdown` in order for the data set to get downloaded as soon as no already existing data set could be found in `/opt/`.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and
depths between 24 and 64 (model.py lines 93-97)

The model includes RELU layers to introduce non-linearity with each
convolution layer (model.py lines 93-97). The data is normalized in the model
using a Keras lambda layer (model.py line 91).

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on vastly different data sets
to ensure that the model was not overfitting (code line 144-153).

The model used the images of a data set that had avoiding overfitting in mind.
The data set's images also used vertically mirrored copies as well. I tried out this approach
deliberately without Dropout as I wanted to see in how far this data set and added
mirroring would have an impact already without dropout. For more details on what the data set looks like, please refer to 4. Appropriate training data.

The model was tested by running it through the simulator and ensuring that the
vehicle could stay on the track, which it did. This was one of those 'lesson learned' moments for me.
The quality of the data set coupled with measures to generalize this
data set even further made dropout more of an option, less of a necessity for this model.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 152).


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving in the center of the road on one hand,
but also to already reduce risk of overfitting on the other. This was to be achieved by a high focus on data diversity. The car only avoiding crashes would not teach the network a good driving experience, but neither would only driving around in perfect circles. This was my approach:

The data set should:

- contain not only left way laps, but also right way laps, as this is like driving a different track
- reduce the amount of left way laps (original direction) from six to three, but add two right way laps
- add a 'best of' of driving maneuvers which would put the car out of a very extreme situation
- add at least one good lap with an entirely different track

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make driving look as safe and coordinated as possible, while not overfitting.

My first step was to use a convolution neural network model similar to the Dave-2 Nvidia model, which I stumbled upon through Paul Heraty's forum post. I then researched it here: [Nvidia Dave-2 model](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)
I thought this model might be appropriate because it was designed with self-driving in mind and - in my opinion - utilized convolution layers very interestingly. That's why I wanted to give this a shot.

In order to gauge how well the model was working, I split my image and steering
angle data into a training and validation set. I then used these sets as inputs
to my various iterative model designs. The very first one I ran on my own hardware.
This prototype model used just a lambda layer, a convolution layer with a 3x3 filter, a flatten layer and two dense layers (with widths of 64 and 1).

Primarily I wanted to see if the pipeline worked, if the model got trained as it should and whether I could deploy it to the simulator.
But it also gave me a lot of insight into how to read and interpret the training and validation loss and in what direction I should go from here.

I found that this first model had a low mean squared error on the training set (around 1.5%)
but a high mean squared error on the validation set (around 30%).
This and the fact that despite low training loss the car drove like the concept of a road didn't exist to it
showed me very directly that the model was overfitting.

To combat the overfitting, I modified the model so that more Convolution layers got added.
Yes, I built versions using dropout layers as well, but with more iterations I got
curious about how the dataset's quality could replace a certain necessity for
applying dropout to reduce overfitting entirely.

Then I set out to record a better data set then the one I used before. I go into detail about that under step *3. Creation of the Training Set & Training Process*

Then I iteratively bumped up the network architecture's complexity, as a too small network would regularly still overfit.
Also I relatively early began to apply cropping to focus the network on what actually is 'important'. This is another one of those 'lesson learned' moments for me: If you can crop, do it.
The final iteration was actually not meant to be the final one, but much rather a test
of a suspicion: I wanted to test out whether the diversification of the data set was done well enough and was present enough
to the network so that dropout layers could become superfluous. And to my surprise, this was the case.

The final step was to run the simulator to see how well the car was driving around the lake side track.
In the beginning it was quite the ride with the first prototype model. It really did not grasp
general concepts of what the environment is, not even speaking about the environment influencing the driving behavior.
To improve the driving behavior in these cases, I used the previously described iterative process: Making sure the network's training loss and validation loss equally get lower on better data, with a deeper network.

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road. The `video.mp4` is the recording of the final model taking over the wheel.

#### 2. Final Model Architecture

Inspired by the Nvidia Dave-2 model, which I stumbled upon through Paul Heraty's forum post,
the final model architecture (model.py lines 89-103) consists of a convolution neural network
with the following layers and layer sizes:

| #  | Layer                | Description                                   |
|:--:|:--------------------:|:---------------------------------------------:|
| 1  | Lambda               | Takes in 160x320x3 images and normalizes them |
| 2  | Cropping             | Crop away upper 50 pixels for unrelated scenery, lower 20 pixels for the hood |
| 3  | Convolution #1  5x5  | 2x2 stride, 24 filters                        |
| 4  | ReLU                 |                                               |
| 5  | Convolution #2  5x5  | 2x2 stride, 36 filters                        |
| 6  | ReLU                 |                                               |
| 7  | Convolution #3  5x5  | 2x2 stride, 48 filters                        |
| 8  | ReLU                 |                                               |
| 9  | Convolution #4  3x3  | 1x1 stride, 64 filters                        |
| 10 | ReLU                 |                                               |
| 11 | Convolution #5  3x3  | 1x1 stride, 64 filters                        |
| 12 | ReLU                 |                                               |
| 13 | Flattening           |                                               |
| 14 | Fully connected #1   | outputs 100                                   |
| 15 | Fully connected #2   | outputs 50                                    |
| 16 | Fully connected #3   | outputs 10                                    |
| 17 | Fully connected #4   | outputs 1                                     |

#### 3. Creation of the Training Set & Training Process

I already hinted at it in step *1. Solution Design Approach*: I focused a lot on building what I considered a diverse and 'rich' data set.

The data set consisted of these recordings:
- three track laps, as perfect as I got it with keyboard input
- two track laps driven in the opposite way, deliberately a bit less perfect but still perfectly fine on the road
- a 'snapshot' lap - 'problematic' areas of the map where I put the car in an extreme position
and then recorded me correcting it
- a lap around the hill road track

An example image `right_2021_09_04_21_42_09_631.jpg` looks like this:

![alt text][image1]

I then also recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to find its way back to a desired way of driving on the track. These images show what an example recovery looks like:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I did an a additional carefully driven lap around the hill road track as well.

![alt text][image6]

To augment the data sat, I also flipped images and angles thinking that this would remove a steering angle bias and add to the diversity of the data set, as practically a new course this way was driven and could be learned from. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

This was done after the practical data collection, but before training (see model.py line 80)

After the collection process, I had 59.166 individual, (not yet flipped) images. The image was read in as is, meaning as RGB image, then mirrored and the mirrored copy was attached to the image set, along with the respective inverted steering angle.

I then randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over- or underfitting. The ideal number of epochs was 3. I found that with a fourth epoch the training set error still would decrease, but the validation error climbed a bit again. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The final training loss was `0.00223`<br>
The final validation loss was `0.02501`
