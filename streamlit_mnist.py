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
    # print("preprocess h,w",h,w,"top left",top,left)
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


def multiple_classification(image):
    gray = np.array(image * 255).astype('uint8')
    cnts, _ = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts.sort(key=lambda c: np.min(c[:, :, 0])) # contour's order is according to x coordinate
    # cv2.drawContours(gray, cnts, -1, (0, 1, 0), 2)
    segment_im = [image]
    for c in cnts:
        mask = np.zeros(gray.shape, dtype='uint8')  # 依Contours圖形建立mask
        cv2.drawContours(mask, [c], -1, 255, -1)  # 255        →白色, -1→塗滿
        segment_im.append(cv2.bitwise_and(image, image, mask=mask))

    if len(segment_im) == 1:
        prediction = 'No Number'
        multi_image = np.zeros((28,28)).astype('float32')
    else:
        x = torch.zeros((len(segment_im)-1,1,28,28))
        multi_image = np.zeros((28, 28*(len(segment_im)-1))).astype('float32')
        for i in range(len(segment_im)-1):
            img = preprocess(segment_im[i+1])
            x[i,0] = torch.tensor(img)
            multi_image[:,i*28:(i+1)*28] = img
        x = x.view(x.size(0), -1)
        W = torch.load('mnist_weight.pth')
        layers = len(W)
        relu = nn.ReLU()
        for i in range(layers - 1):
            if i == 0:
                newx = W[i](x)
            else:
                newx = W[i](newx)
            newx = relu(newx)
        yy = W[layers - 1](newx)
        # print("yy",yy.shape)
        predict = torch.argmax(F.softmax(yy, dim=1),dim=1).tolist()
        prediction = str(predict)[1:-1]
    return multi_image, prediction


def Testing(x):
    img = preprocess(x)
    x = torch.tensor(img)
    x = x.unsqueeze(0).unsqueeze(0)
    x = x.view(x.size(0),-1)
    W = torch.load('mnist_weight.pth')
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
