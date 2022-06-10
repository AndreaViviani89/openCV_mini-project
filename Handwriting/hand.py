import cv2 as cv
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import torch.nn as nn

def load_images(file_pth, show=False):

    digits_pth = file_pth
    digits = cv.imread(digits_pth)

    gray_digits = cv.cvtColor(digits, cv.COLOR_BGR2GRAY)

    if show:
        plt.figure(figsize=(15, 8))
        plt.imshow(gray_digits, cmap='gray')
        plt.show()

    return digits, gray_digits

def my_model():
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size),
                          nn.LogSoftmax(dim=1))
    return model

def preprocess(img, n_digits, offset=70, thresh=100):
    # binarize image
    _, threshold = cv.threshold(img, thresh, 255, cv.THRESH_BINARY_INV)

    # dilate to better find contours
    dilated = cv.dilate(threshold, (3, 3), iterations=5)

    # find contours
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
    final_contours = sorted_contours[:n_digits]
