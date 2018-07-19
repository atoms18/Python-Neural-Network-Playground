
import math

import keras
from keras import backend

from extra_activation import ExtraActivation
from datasets import Datasets
import app

DEFAULT_WIDTH = 1366
DEFAULT_HEIGHT = 768

DATASETS_NUM = 500
TEST_DATASETS_RATIO = 0.2

outputPanelSize = 300
outputPanelRange = [-6, 6]
outputCanvasDensity = 100

DEFAULT_MODEL = {
    "inputs": ["x", "y"],
    "hiddens": [
        {
            "units": 8,
            "activation": "Tanh"
        },
        {
            "units": 4,
            "activation": "Tanh"
        },
        {
            "units": 2,
            "activation": "Tanh"
        },
    ],
    "outputs": {
        "activation": "Tanh"
    }
}

INPUTS = {
    "x": lambda x, y: x,
    "y": lambda x, y: y,
    "xy": lambda x, y: x * y,
    "sin(x)": lambda x, y: math.sin(x),
    "sin(y)": lambda x, y: math.sin(y),
    "cos(x)": lambda x, y: math.cos(x),
    "cos(y)": lambda x, y: math.cos(y),
}

ACTIVATION_FUNC_LIST = {
    "Tanh": backend.tanh,
    "Sigmoid": backend.sigmoid,
    "ReLU": backend.relu,
    "Gaussian": ExtraActivation.gaussian,
    "Sin": ExtraActivation.sin
}

DATASETS_LIST = {
    "Circle": Datasets.circle,
    "Line": Datasets.line,
    "And": Datasets.and_,
    "Exclusive OR": Datasets.xor,
    "Spiral": Datasets.spiral,
    "Plane Regression": Datasets.plane_regression,
    "Line Regression": Datasets.line_regression,
}

LEARNING_RATE_LIST = [
    1e-5,
    1e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1,
    3
]

OPTIMIZERS_LIST = {
    "SGD": keras.optimizers.SGD,
    "RMSprop": keras.optimizers.RMSprop,
    "Adagrad": keras.optimizers.Adagrad,
    "Adadelta": keras.optimizers.Adadelta,
    "Adam": keras.optimizers.Adam,
    "Adamax": keras.optimizers.Adamax,
    "Nadam": keras.optimizers.Nadam,

}

# (output range)
# 0 is for [-1 -> 1]
# 1 is for [0 -> 1]
# 2 is for [0 -> 1, 0 -> 1]
# But you have to uncomment
# some code at app.py:mapOutputToRGBA() for this to work
# because there are lag if I'm not commenting those
MODE = 1
DEBUG = False
DATASETS_FORM = {"train": [], "test": []}
TRAIN_DATASETS_FORM = {
    "X_train": [], "Y_train": [], "X_test": [], "Y_test": []}

if __name__ == '__main__':
    mainApp = app.MainApp()
    mainApp.MainLoop()
    mainApp.model.stopTraining()
