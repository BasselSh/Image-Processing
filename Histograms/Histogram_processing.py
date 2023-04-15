# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:21:23 2023

@author: Bassel
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class Image:
    id_plot=0
    path_input = None
    path_output = None
    def __init__(self, img=None,histSize=256, histRange=(0,256), CONFIG="BGR", NORMALIZE = True, pth = "default", sequence = True):
        self.img_on = None
        self.img_org = None
        self.sequence = sequence
        self.img_org_hist = None
        self.history = []
        self.history_hist = []
        self.history_hist_figure = []
        self.operations = []
        self.n_operations = 0
        if pth == "default":
            path = os.getcwd()
        else:
            path = pth
        Image.path_input = os.path.join(path,"inputs")
        Image.path_output = os.path.join(path , "outputs")
        self.last_executed = ""
        self.histSize=256
        self.NORMALIZE = NORMALIZE
        if CONFIG=="BGR":
            a=[0,1,2]
        else:
            a=[2,1,0]
        self.order=a
        Image.histSize=histSize
        Image.histRange=histRange
        if img is not None:
            self.set_img(img)
        else:
            self.img=None
    
    def copy_img(self):
        if self.sequence == False:
            return self.img_on.copy()
        else:
            return self.img.copy()        
    
    def get_img(self):
        return self.img
    
    def set_img(self, img, text = None):
        self.n_operations = self.n_operations +1
        if img is None:
            return
        if text is not None:
            print(text)
        else:
            text = str(self.n_operations)
        if self.img_org is None:
            print("added origin")
            self.img_org = img
            self.img_on = img
        self.img = img
        self.rows, self.cols = self.img.shape[0:2]
        self.history.append(self.img)
        self.operations.append(text)
        self.calc()
        
        
    def on_current(self):
        self.img_on = self.img
        self.sequence = False
    def on_sequence(self):
        self.sequence = True
        
    def calc(self,img=None):
        print("Calculating Histogram")
        if img is None:
            if self.img is None:
                print("error")
                return
            img=self.img
        img_s=cv2.split(img)
        bHist=cv2.calcHist(img_s,[self.order[0]],None, [self.histSize], (0, 256))
        gHist=cv2.calcHist(img_s,[self.order[1]],None, [self.histSize], (0, 256))
        rHist=cv2.calcHist(img_s,[self.order[2]],None, [self.histSize], (0, 256))
        self.img=img
        self.bH=bHist
        self.gH=gHist
        self.rH=rHist
        if self.NORMALIZE:
            self.normalize()
        else:
            self.history_hist.append((self.bH, self.gH, self.rH))
     
    def normalize(self):
        if self.last_executed == "":
            self.last_executed = "normalized"
        print("Normalizing")
        
        self.bH_not_normalized = self.bH
        self.gH_not_normalized = self.gH
        self.rH_not_normalized = self.rH
        max_b = np.sum(self.bH)
        max_g = np.sum(self.gH)
        max_r = np.sum(self.rH)
        self.bH = self.bH/max_b
        self.gH = self.gH/max_g
        self.rH = self.rH/max_r
        self.NORMALIZE = True
        self.history_hist.append((self.bH, self.gH, self.rH))
        if self.img_org_hist is None:
            self.img_org_hist = (self.bH, self.gH, self.rH)
        
        
    def show(self, image = "current", name=None):
        breakpoint()
        if image == "org":
            I = self.img_org
            
        else:
            I = self.img
            
        if name is None:
            name="number"+str(Image.id_plot)
        image_name = name + "_" + self.last_executed
        image_path = Image.path_output + "/" + "images"
        hist_path = Image.path_output + "/" + "Histograms"
        
        try:
            os.mkdir(image_path)
        except IOError:
            pass
        try:
            os.mkdir(hist_path)
        except IOError:
            pass
        
        self.__show_hist(self.history_hist[-1])
        plt.suptitle(name)
        #plt.savefig(hist_path + "/" + image_name  + "_Histogram" + ".png")
        plt.show()
        self.__show_img(I)
        #plt.imsave(image_path + "/" + image_name + ".jpg", I)
        
    def __show_img(self, I):
        Image.id_plot=Image.id_plot+1
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        fig = plt.figure(Image.id_plot)
        
        plt.imshow(I)
        plt.show()
    def __show_hist(self, hist_tuple):
        bH, gH, rH = hist_tuple
        Image.id_plot=Image.id_plot+1
        fig = plt.figure(Image.id_plot)
        self.history_hist_figure.append(fig)
        t=range(256)
        plt.plot(t,bH, color="blue")
        plt.plot(t,gH, color="green")
        plt.plot(t,rH, color="red")
        
    def show_original(self):
        self.show(image = "org")
        
    def show_history(self):
        #Image.id_plot=Image.id_plot+1
        for i in range(len(self.history)):
            self.__show_hist(self.history_hist[i])
            plt.show()
            self.__show_img(self.history[i])
            plt.show()
    def save_history(self, folder = None):
        
        if len(self.operations) == 0:
            names = list(map(str,list(range(len(self.history)))))
        else:
            names = self.operations
        
        if folder is None:
            image_path = Image.path_output + "/" + "History"
        else:
            image_path = Image.path_output + "/" + "History/" + folder  
        
        try:
            os.mkdir(image_path)
        except IOError:
            pass
        ignor = 0
        length = len(self.history)-ignor
        rows, cols = self.history[0].shape[0:2]
        n_cols = 3
        n_rows = int(np.ceil(length/n_cols))
        
        mat = np.ones((rows*n_rows, cols*n_cols, 3 ), dtype = np.uint8)*255
        hist_rows, hist_cols = (288, 432)
        hist_mat = np.ones((hist_rows*n_rows, hist_cols*n_cols, 3), dtype = np.uint8)*255
        Image.id_plot=Image.id_plot+1
        j = 0
        k = 0
        
        for i in range(len(self.history)):
            plt.figure(self.history_hist_figure[i])
            plt.savefig(image_path  + "/"   + names[i] + "_Histogram" + ".png")
            hist = cv2.imread(image_path  + "/"   + names[i] + "_Histogram" + ".png")
            hist = cv2.cvtColor(hist, cv2.COLOR_BGR2RGB)
            I = cv2.cvtColor(self.history[i], cv2.COLOR_BGR2RGB)
            plt.imsave(image_path + "/" + names[i] + ".jpg", I)
            if i>=ignor:
                mat[k*rows:rows*(k+1), j*cols:cols*(j+1),:] = I
                hist_mat[k*hist_rows:hist_rows*(k+1), j*hist_cols:hist_cols*(j+1),:] = hist
                if j ==(n_cols-1):
                    j=-1
                    k = k+1
                j = j +1
        plt.imsave(image_path + "/ALL_images" + ".jpg", mat)
        plt.imsave(image_path + "/ALL_images_histogram" + ".jpg", hist_mat)
        plt.imshow(mat)
        plt.imshow(hist_mat)
        
def profile(img, x):
    return img[x,:]

def project_(img,xy):
    return np.sum(img,xy)/(img.shape[(xy+1)%2])



if __name__ == "__main__":
    pass