import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def preprocess(image):
    top = 0
    left = 0
    bottom = image.shape[0] - 1
    right = image.shape[1] - 1
    for i in range(image.shape[0]):
        if image[i, :].sum() > 0:
            top = i
            break
    for j in range(image.shape[1]):
        if image[:, j].sum() > 0:
            left = j
            break
    for i in range(image.shape[0] - 1, -1, -1):
        if image[i, :].sum() > 0:
            bottom = i
            break
    for j in range(image.shape[1] - 1, -1, -1):
        if image[:, j].sum() > 0:
            right = j
            break
    # print("top is", top, "bottom is", bottom, "left is", left, "right is", right)
    image1 = image[top:bottom + 1, left:right + 1]
    h = bottom - top + 1
    w = right - left + 1
    if h > w:
        h1 = 18
        w1 = int(w * 18 / h)
    else:
        w1 = 18
        h1 = int(h * 18 / w)
    image2 = cv2.resize(image1, (w1, h1))
    ini_x = int((28 - h1) / 2)
    ini_y = int((28 - w1) / 2)
    image3 = np.zeros((28, 28)).astype('float32')
    image3[ini_x:ini_x + h1, ini_y:ini_y + w1] = image2
    return image3

def Testing(x):
    img = preprocess(x)
    x = torch.tensor(img)
    x = x.unsqueeze(0).unsqueeze(0)
    # print('unsqueeze',x.shape)
    x = x.view(x.size(0),-1)
    W = torch.load('mnist_weight.pth')
    # print(W[0].weight.shape,W[0].bias.shape,x.shape)
    layers = len(W)
    relu = nn.ReLU()
    for i in range(layers-1):
        if i == 0:
            newx = W[i](x)
        else:
            newx = W[i](newx)
        newx = relu(newx)
    yy = W[layers-1](newx)
    prediction = F.softmax(yy, dim=1)
    return img, prediction
