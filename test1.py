import os
import shutil
import glob
from shutil import copyfile
import numpy as np
import time

htmlfiles = [os.path.join(root, name)
             for root, dirs, files in os.walk('mfccnpyFiles')
             for name in files
             if name.endswith((".npy"))]
htmlfiles1=[os.path.join(root, name)
             for root, dirs, files in os.walk('phonemeLabels')
             for name in files
             if name.endswith((".npy"))]

count2=0
last_time = time.time()
for file in htmlfiles:
    fullFilename = os.path.join(file)
    #print(fullFilename)
    filenameNoSuffix =  os.path.splitext(fullFilename)[0]
    count=0
    print (filenameNoSuffix.split('/')[-1])
    for file1 in htmlfiles1:
        fullFilename1 = os.path.join(file1)
        filenameNoSuffix1 =  os.path.splitext(fullFilename)[0]
        if(filenameNoSuffix1.split('/')[-1]==filenameNoSuffix.split('/')[-1]):
            #print(filenameNoSuffix1.split('/')[-1])
            count+=1
            print (filenameNoSuffix1.split('/')[-1])
            print("*************Input data**************")
            input_val = np.load("./mfccnpyFiles/" +filenameNoSuffix.split('/')[-1]+'.npy')
            print (np.shape(input_val))
            print("Output phoneme val:*****************")
            output_val=np.load("./phonemeLabels/" +filenameNoSuffix.split('/')[-1]+'.npy')
            print(np.shape(output_val))
            count2=count2+1
            print (count2)
            if(count>=1):
                break

new_time = time.time()
print ("Total time taken is :",(new_time-last_time))

