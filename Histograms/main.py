# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 20:06:16 2023

@author: Bassel
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from Histogram_processing import Image
from transformations import transformation
#__all__ = ['Histogram_processing', 'transformation']







def main1():
    path = os.getcwd()
    path_input = os.path.join(path,"inputs")
    path_output = os.path.join(path , "outputs")
    I=cv2.imread(path_input + '/dark_sky.jpg')
    I=cv2.resize(I,(500,500))
    img=I.copy()
    img=cv2.resize(img,(500,500))  
    #Parameters to control histogram
    histSize = 256
    histRange = (0, 256)


    #histogram of the image
    H00=Image(I)
    H00.equalize()
    H00.show("original")


    #Histogram of the shifted image
    shift_amounts=range(10,240,40)
    for shift_amount in shift_amounts:
        H00=Image(I)
        H00.shift(amount= shift_amount)
        H00.show("after shift"+str(shift_amount))

    #extend with low frequencies
    alphas = range(10,1, -2)
    for alpha in alphas:
        alpha = alpha/10
        H2=Image(I)
        H2.extend(alpha = alpha, REMOVE_LOW_FREQUENCY = False)
        H2.show("Extending")

    #extend with low frequencies filtering
    H2=Image(I)
    H2.extend(REMOVE_LOW_FREQUENCY=True)
    H2.show("Removing low frequencies")

def main2():
    path = os.getcwd()
    path_input = os.path.join(path,"inputs")
    path_output = os.path.join(path , "outputs")



    I=cv2.imread(path_input + '/dark_sky.jpg')
    I=cv2.resize(I,(500,500))
    img=I.copy()
    img=cv2.resize(img,(500,500))  

    ob = transformation(on_origin = True, img = I)
    ob.rotate(45)
    ob.extend()
    ob.shift()
    ob.sinusoid()
    ob.piecewise()
    ob.poly()
    ob.barrel()
    ob.debarrel()
    ob.stitching()
    ob.show_history()

if __name__ =="__main__":
    main2()