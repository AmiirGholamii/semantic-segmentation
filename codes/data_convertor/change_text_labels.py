import os
import cv2
import numpy as np
directory = "/home/rider/DataSets/Images/Development/humanoid_soccer_dataset/ScreenshotMasks"
for filename in os.listdir(directory):
    if filename.endswith(".txt"): 
        blank_image = np.zeros((480,640), np.uint8)
        with open(os.path.join(directory, filename)) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                splitted_list = lines[i].split(' ')
                for j in range(len(splitted_list)-1):
                    blank_image[i][j] = (splitted_list[j])
        cv2.imwrite(os.path.join(directory, filename.replace(".txt",".png")),blank_image)
        cv2.waitKey(0)
        # print(os.path.join(directory, filename))
        continue
    else:
        continue
