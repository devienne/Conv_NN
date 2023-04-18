#!/usr/bin/env python
# coding: utf-8

# # Data Augmentation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from numpy import genfromtxt


def gaussian_noise(img, sigma, mean=0):
    img = data.copy()
    noise = np.random.normal(mean, sigma, img.shape)
    mask_overflow_upper = img+noise >= 1.0
    mask_overflow_lower = img+noise <= -1.0
    noise[mask_overflow_upper] = 1.0
    noise[mask_overflow_lower] = -1.0
    img += noise
    return img


filepath = '/Users/Jose/Desktop/Big_Data/Project_2/Seismic_Data/train_earthquake'

for file in os.listdir(filepath):
    dir = filepath + '/' + file
    if os.path.isdir(dir) == True:
        for file2 in os.listdir(dir):            
            data = genfromtxt('{}/{}'.format(dir, file2), delimiter=',') 
            data = data[data < 1] # to guaratee that there's no values greater than 1 
            for sigma in np.arange(2.0*10**-4,3.1*10**-4, 0.1*10**-4):
                aug = gaussian_noise(data, sigma)
                aug = aug[aug < 1] # # to guaratee that there's no values greater than 1
                filename = os.path.splitext(file2)[0]
                pd.DataFrame(aug).to_csv('{}/{}_{}_aug.csv'.format(dir, filename,sigma))            
    

