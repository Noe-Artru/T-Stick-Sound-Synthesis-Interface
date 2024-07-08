## The model takes in the inputs from the OSC data of the TStick and outputs synthesis parameters
## The goal is to create interesting and new sounds that the user can understand and manipulate
## The issue of such a model is creating complex relationships between the inputs and outputs that are too scrambled to be understood by a user
## This means the model should have a modified loss function that penalizes inputs being used for too many different outputs, and outputs having too many associated inputs
## I don't have any training data -> make it by hand and then interpolate in feature space ?

## First idea -> Make the feature space 2D, so the user can create data points
## Second Idea -> Make the model recognise certain gestures that the user makes and map them to certains sounds

import torch
import torch.nn as nn

class DecoderNN(nn.Module):
    def __init__(self):
        super(DecoderNN, self).__init__()
        self.fc1 = nn.Linear(2, 30)
        self.fc2 = nn.Linear(30, 50)
        self.fc3 = nn.Linear(50, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 20)
        self.fc6 = nn.Linear(20, 9)
        self.sliderRangeMax = torch.Tensor([1000, 3, 3, 10, 0.1, 1, 2, 2, 0.4])
        self.sliderRangeMin = torch.Tensor([20, 0.1, 0.1, 0, 0, 0.01, 0, 0.01, 0])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        x = self.sliderRangeMin + (self.sliderRangeMax - self.sliderRangeMin) * x
        return x
