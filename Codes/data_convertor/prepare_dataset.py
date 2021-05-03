import glob
import os
data_path = "/home/rider/DataSets/Images/Screenshots"
mask_path = "/home/rider/DataSets/Images/Development/ScreenshotMasks"
filelist = glob.glob(os.path.join(data_path, '*.png'))
masklist = glob.glob(os.path.join(mask_path, '*.txt'))
count = 0
for infile in sorted(filelist): 
  count=count + 1
  #do some fancy stuff
  os.rename(infile, data_path+"/"+str(count)+".png") 
for infile in sorted(masklist): 
  count=count + 1
  #do some fancy stuff
  os.rename(infile, mask_path+"/"+str(count)+".txt") 