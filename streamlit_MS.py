# Project page of IFCNN is https://github.com/uzeful/IFCNN.

import os
import cv2
import time
import torch
import torch.utils.data as data
import sys
# sys.path.append(r'D:\nspoDNN\venv\IFCNN\Code\streamlit_app')
from model import myIFCNN

# os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']='0'

from torchvision import transforms
from torch.autograd import Variable

from PIL import Image
import numpy as np
import math
from myTransforms import denorm, norms, detransformcv2


def makeTestingData(block_size, block_extend, img):
    # print(route_blur)
    # img = cv2.imread(route_blur)
    # img = np.array(Image.open(route_blur))
    # print(img.dtype, img.max())
    # img = np.array(img).astype('float32')
    # print(img.dtype, img.max())
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w, c = img.shape
    # block_numM = int(h / block_size)
    # block_numN = int(w / block_size)
    block_numM = math.ceil(h/block_size)
    block_numN = math.ceil(w/block_size)
    m1 = int((block_numM*block_size-h)/2)
    m2 = block_numM*block_size-h-m1
    n1 = int((block_numN*block_size-w)/2)
    n2 = block_numN*block_size-w-n1
    h1 = block_numM*block_size
    w1 = block_numN*block_size
    imgEx = cv2.copyMakeBorder(img, m1, m2, n1, n2, cv2.BORDER_REFLECT_101)
    # imgEx = cv2.copyMakeBorder(img, block_extend, block_extend, block_extend, block_extend, cv2.BORDER_REFLECT_101)
    arrayBlur = []
    for ii in range(0, h1 - block_size + 1, block_size):
        for jj in range(0, w1 - block_size + 1, block_size):
            x = imgEx[ii:ii + block_size+block_extend*2, jj:jj + block_size+block_extend*2, :]
            arrayBlur.append(x)
    arrayBlur = np.array(arrayBlur)
    # arrayBlur = arrayBlur.astype('uint8')
    return arrayBlur, block_numM, block_numN, h, w, m1, n1


class ImagePair(data.Dataset):
    def __init__(self, impatch1, impatch2, mode='RGB', transform=None):
        self.impatch1 = impatch1
        self.impatch2 = impatch2
        self.mode = mode
        self.transform = transform

    def get_pair(self):
        img1 = self.impatch1
        img2 = self.impatch2
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2


def reconImg(arrayClear, block_numM, block_numN, block_size, block_extend, h, w, m1, n1):
    h1 = block_size * block_numM
    w1 = block_size * block_numN
    img = np.zeros((h1, w1, 3)).astype('float32')
    index = np.zeros((h1, w1, 3)).astype('float32')
    count = 0
    for ii in range(0, h1 - block_size + 1, block_size):
        for jj in range(0, w1 - block_size + 1, block_size):
            img[ii:ii + block_size, jj:jj + block_size, :] += arrayClear[count][block_extend:block_extend+block_size,
                                                           block_extend:block_extend+block_size, :]
            index[ii:ii + block_size, jj:jj + block_size, :] += 1
            count += 1
    img = np.divide(img, index)
    img = img[m1:m1+h,n1:n1+w,:]
    return img

device = torch.device("cpu")
# device = torch.device("cuda:0")
def doing_fusion(image1, image2):
    fuse_scheme = 0
    if fuse_scheme == 0:
        model_name = 'IFCNN-MAX'
    elif fuse_scheme == 1:
        model_name = 'IFCNN-SUM'
    elif fuse_scheme == 2:
        model_name = 'IFCNN-MEAN'
    else:
        model_name = 'IFCNN-MAX'

    # load pretrained model
    sigma = 0.02
    model = myIFCNN(fuse_scheme=fuse_scheme)
    model.load_state_dict(torch.load('sigma' + str(sigma) + '_epoch500.pth'))
    # model.load_state_dict(torch.load(r'D:\nspoDNN\venv\IFCNN\Code\snapshots\IFCNN-MAX.pth'))
    model.eval()
    model = model.to(device)

    # test_set=['FS5_G000_MS_L1A_20191122_025853','FS5_G000_MS_L1A_20210308_025920','FS5_G000_MS_L1A_20210408_034832',
    #           'FS5_G000_MS_L1A_20210727_052602','FS5_G053_MS_L1A_20210113_030314','FS5_G054_MS_L1A_20210113_030317']
    # test_index = 3
    # route1 = r'D:\nspoDNN\venv\FS5_image/'+test_set[test_index]+'_NSPO.bmp'
    # route2 = r'D:\nspoDNN\venv\FS5_image/'+test_set[test_index]+'_NTHUEE.bmp'
    route_fuse = r'IFCNN_streamlit.bmp'


    block_size = 200
    block_extend = 0
    patch1, block_numM, block_numN, h, w, m1, n1 = makeTestingData(block_size, block_extend, image1)
    patch2, block_numM, block_numN, h, w, m1, n1 = makeTestingData(block_size, block_extend, image2)
    arrayFuse = []
    begin_time = time.time()
    for ind in range(patch1.shape[0]):
        is_gray = True  # Color (False) or Gray (True)
        mean = [0, 0, 0]  # normalization parameters
        std = [1, 1, 1]
        # block1 = np.expand_dims(patch1[ind], axis=2).repeat(3, axis=2)
        # block2 = np.expand_dims(patch2[ind], axis=2).repeat(3, axis=2)
        block1 = np.array(patch1[ind])
        block2 = np.array(patch2[ind])

        # load source images
        pair_loader = ImagePair(impatch1=block1, impatch2=block2, transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]))
        img1, img2 = pair_loader.get_pair()
        # print(img1.mean(), img2.mean())
        img1.unsqueeze_(0)
        img2.unsqueeze_(0)

        # perform image fusion
        with torch.no_grad():
            res = model(Variable(img1.to(device)), Variable(img2.to(device)))
            res = denorm(mean, std, res[0]).clamp(0, 1)
            res_img = res.cpu().data.numpy()
            img = res_img.transpose(1,2,0)
        arrayFuse.append(img)
    imgFuse = reconImg(arrayFuse, block_numM, block_numN, block_size, block_extend, h, w, m1, n1)
    # imgFuse = np.clip(imgFuse, 0, 1)
    # print(route_fuse)
    # cv2.imwrite(route_fuse, (imgFuse*255).astype('uint8'))
    imgFuse = cv2.cvtColor(imgFuse, cv2.COLOR_BGR2RGB)
    stop_time = time.time()
    proc_time = stop_time - begin_time
    print('Total processing time : {:.3}s'.format(proc_time))
    return imgFuse, proc_time



