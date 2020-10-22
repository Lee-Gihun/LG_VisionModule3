import numpy as np
import os
import cv2

__all__=['_get_mean', '_get_var']

def _get_mean(path):
    tmps = os.listdir(path)
    
    R_channel = 0
    G_channel = 0
    B_channel = 0

    total_pixel = 0
    for tmp in tmps:
        PATH = os.path.join(path, tmp)
        images = os.listdir(PATH)
        for image in images:
            img = cv2.imread(os.path.join(PATH, image))
            # print (img.shape)
            img = img / 255
            total_pixel += img.shape[0] * img.shape[1]

            R_channel += np.sum(img[:, :, 0])
            G_channel += np.sum(img[:, :, 1])
            B_channel += np.sum(img[:, :, 2])

    R_mean = R_channel / total_pixel
    G_mean = G_channel / total_pixel
    B_mean = B_channel / total_pixel
    
    return R_mean, G_mean, B_mean

def _get_var(path):
    tmps = os.listdir(path)
    R_mean, G_mean, B_mean = _get_mean(path)
    
    R_channel = 0
    G_channel = 0
    B_channel = 0
    total_pixel = 0
    
    for tmp in tmps:
        PATH = os.path.join(path, tmp)
        images = os.listdir(PATH)
        for image in images:
            #print(image)
            img = cv2.imread(os.path.join(PATH, image))
            # print (img.shape)
            img = img / 255
            total_pixel += img.shape[0] * img.shape[1]

            R_channel += np.sum((img[:, :, 0] - R_mean) ** 2)
            G_channel += np.sum((img[:, :, 1] - G_mean) ** 2)
            B_channel += np.sum((img[:, :, 2] - B_mean) ** 2)

    R_std = np.sqrt(R_channel / total_pixel)
    G_std = np.sqrt(G_channel / total_pixel)
    B_std = np.sqrt(B_channel / total_pixel)
    
    return R_std, G_std, B_std