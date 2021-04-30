import cv2
import glob
import os
import numpy as np
hasty_labels_path = "/home/amir/enel645-project-master/Test/label"
new_labels_path = "/home/amir/enel645-project-master/Test/Newlabel"
filelist = glob.glob(os.path.join(hasty_labels_path, '*.png'))
for infile in filelist: 
    label_img = cv2.imread(infile)
    blank_image = np.zeros((480,640), np.uint8)
    for i in range(len(label_img)):
        for j in range(len(label_img[0])):
            if tuple(label_img[i][j]) == (232,144,69):
                blank_image[i][j] = 0 #background
            elif tuple(label_img[i][j]) == (25,176,106):
                blank_image[i][j] = 1 #field  
            elif tuple(label_img[i][j]) == (255,255,255):
                blank_image[i][j] = 2 #line
            elif tuple(label_img[i][j]) == (180,120,31):
                blank_image[i][j] = 3 #ball
            elif tuple(label_img[i][j]) == (235,62,156):
                blank_image[i][j] = 4 #robot
            elif tuple(label_img[i][j]) == (28,26,227):
                blank_image[i][j] = 5 #goal
            else:
                blank_image[i][j] = 0 #background
    cv2.imwrite(os.path.join(new_labels_path,os.path.basename(infile)),blank_image)

