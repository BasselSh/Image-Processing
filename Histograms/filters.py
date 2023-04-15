# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:27:54 2023

@author: Bassel
"""
from Histogram_processing import Image
from skimage.util import random_noise
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class filters(Image):
    def __init__(self, img=None,histSize=256, histRange=(0,256), CONFIG="BGR", NORMALIZE = True, pth = "default", sequence = True ):
        super().__init__(img ,histSize, histRange, CONFIG, NORMALIZE, pth, sequence)
        self.window_size = 3
    
    def noise_saltnpepper(self):
        I = self.copy_img()
        I_noise = random_noise(I, mode = 's&p')
        I_noise = np.clip(255*I_noise, 0, 255)
        I_noise = np.asarray(I_noise, dtype = np.uint8)
        self.set_img(I_noise, "pepper_noise")
    
    def noise_gaussian(self):
        I = self.copy_img()
        I_noise = random_noise(I, mode = "gaussian")
        I_noise = np.clip(255*I_noise, 0, 255)
        I_noise = np.asarray(I_noise, dtype = np.uint8)
        self.set_img(I_noise, "gaussian_nosie")
    
    def noise_speckle(self):
        I = self.copy_img()
        I_noise = random_noise(I, mode = 'speckle')
        I_noise = np.clip(255*I_noise, 0, 255)
        I_noise = np.asarray(I_noise, dtype = np.uint8)
        self.set_img(I_noise, "speckle_nosie")
    
    def noise_poisson(self):
        I = self.copy_img()
        I_noise = random_noise(I, mode = 'poisson')
        I_noise = np.clip(255*I_noise, 0, 255)
        I_noise = np.asarray(I_noise, dtype = np.uint8)
        self.set_img(I_noise, "poisson_noise")
    
    def counterharmonic_mean_filter(self, Q=1):
        I = self.copy_img().astype(np.float64)
        size=(3,3)
        kernel = np.full(size, 1.0)
        
        num = np.power(I, Q + 1, where = I!=0)
        denum = np.power(I, Q, where = I!=0)

        denum_filtered = cv2.filter2D(denum, -1, kernel)
        num_filtered = cv2.filter2D(num, -1, kernel)

        result =  np.where(denum_filtered == 0, 0, num_filtered/denum_filtered)
        Iout = np.asarray(result, dtype = np.uint8)
        self.set_img(Iout, "coutnerHarmonic" + "Q"+ str(Q))
    
    def gaussian_filter(self, sigma = 1):
        I = self.copy_img()
        size = 6*sigma + 1
        size = 7
        #size = 3
        Iout = cv2.GaussianBlur(I, (size,size), sigmaX = sigma, sigmaY = sigma)
        self.set_img(Iout)
        
    def __adapt(self, I, i, j, max_size):
        size = 3
        #I = I.astype(np.float64)
        rows, cols = I.shape[0:2]
        med = 0
        while True:
            
            k = int((size-1)/2)
            if size >= max_size or (i-k)<0 or (i+k) >= rows or (j-k) < 0 or (j+k) >= cols:
                return med
            crop= I[i-k:i+k+1, j-k:j+k+1]
            Imax = crop.max()
            Imin = crop.min()
            med = np.median(crop)
            A1 = med - Imin
            A2 = med - Imax
            if A1>0 and A2<0:
                Z1 = I[i,j] - Imin
                Z2 = I[i,j] - Imax
                if Z1>0 and Z2<0:
                    return I[i,j]
                else:
                    return med
            else:
                size = size + 2
    def adaptive_median(self):
        I = self.copy_img().astype(np.float64)
        Ib = I[...,0]
        Ig = I[...,1]
        Ir = I[...,2]
        IbOut = Ib
        IgOut = Ig
        IrOut = Ir
        max_size = 11
        half_w = int((max_size-1)/2)
        for i in range(1, I.shape[0] - 1,1):
            for j in range(1, I.shape[1] -1,1):
                print("i,j=", i, j)
                IbOut[i,j] = int(self.__adapt(Ib, i, j, max_size))
                IgOut[i,j] = int(self.__adapt(Ig, i, j, max_size))
                IrOut[i,j] = int(self.__adapt(Ir, i, j, max_size))
        IbOut = IbOut.astype(np.uint8)
        IgOut = IgOut.astype(np.uint8)     
        IrOut = IrOut.astype(np.uint8)
        Iout = cv2.merge([IbOut, IgOut, IrOut])
        self.set_img(Iout, "adaptive")
    
    def __geo(self, I,r,i,j,size):
        mat = I[i-r:i+r+1, j-r:j+r+1]
        return int(255*np.power(np.prod(mat), 1/(size*size)))
    
    def geometric_mean(self):
        I = self.copy_img().astype(np.float64)
        Iout= I.copy().astype(np.uint8)
        I = I/255
        Ib = I[...,0]
        Ig = I[...,1]
        Ir = I[...,2]
        rows, cols = I.shape[0:2]
        size  = 3
        vl = 0
        r = int((size-1)/2)
        for i in range(r+vl, rows-r-vl):
            for j in range(r+vl, cols-r-vl):
                Iout[i,j,0] = self.__geo(Ib,r,i,j,size)
                Iout[i,j,1] = self.__geo(Ig,r,i,j,size)
                Iout[i,j,2] = self.__geo(Ir,r,i,j,size)
        Iout = Iout.astype(np.uint8)
        self.set_img(Iout, "Geometric_mean")
             
        
if __name__ == "__main__":
    path = os.getcwd()
    path_input = os.path.join(path,"inputs")
    path_output = os.path.join(path , "outputs")



    I=cv2.imread(path_input + '/money.jpg')
    ob = filters(I, sequence = True)
    
    ###Applying noise###
    n = 2
    if n == 1:
        ob.noise_gaussian()
    if n == 2:
        ob.noise_saltnpepper()
    if n == 3:
        ob.noise_poisson()
    if n == 4:
        ob.noise_speckle()
    
    ###Apply on the previous image###
    ob.on_current()
    
    ######Removing Noise######
    c = 5
    if c == 1:
        for Q in range(-3, 4):
            ob.counterharmonic_mean_filter(Q)
    if c == 2:   
        for i in range(1, 8, 2):
            sigma = i/10
            ob.gaussian_filter(sigma) 
    if c == 3:
        ob.adaptive_median()
    if c == 4:    
        ob.geometric_mean()
    if c == 5:
        ob.wiener()
    
    ####plotting####
    ob.show_history()
    
    ob.save_history("adaptive_pepper")


    
