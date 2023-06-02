# Image Processing
 

Image Class description

Adding image happens during intializing the class, or by using self.set_img()
When we apply mutliple processings on images, sometime we need to apply the processing on the image resulted from the previous processing, while in other cases, we want to applying processing separately for checking differences. To define that you can set the value Sequence to either true or false.

To get the image which we want to use: self.copy_img()

To identify which image we want to use to apply processing, use self.on_current before calling the function self.copy_img

To add an image, self.set_img()
To show all processing that you have done on the images, self.show_history
To save figures, self.save_history
