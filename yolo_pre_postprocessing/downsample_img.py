import numpy as np
import os
import cv2 as cv

path = "ML_pipeline/raw_data_copy3/"
files = os.listdir(path)

new_files = []
for file in files:
    if file.endswith(".jpeg"):
        new_files.append(file)
        
i =0
for file in new_files:
    print("i={}".format(i))
    i += 1
    img = cv.imread(path+file)
    downsampled = cv.resize(img,(416,416))
    cv.imwrite("ds"+file,downsampled)
    