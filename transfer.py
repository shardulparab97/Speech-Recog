import os
import shutil
import glob
from shutil import copyfile


'''for files in os.walk('TIMIT/TRAIN/'):
    for name in files:
     for x in name:
         if x.endswith('.WAV'):
            dst=os.path.abspath( x)
            print(dst)
            #print(dst.split('/')[-1])
            #print(os.path.abspath(x))
            #copyfile(os.path.abspath(x), "./WAVFiles/"+dst.split('/')[-1])'''

'''or dirpath, dirnames, filenames in os.walk(".TIMIT/TRAIN"):
    for filename in [f for f in filenames if f.endswith(".WAV")]:
        print (os.path.join(dirpath, filename))'''

htmlfiles = [os.path.join(root, name)
             for root, dirs, files in os.walk('TIMIT')
             for name in files
             if name.endswith((".WAV"))]


for file in htmlfiles:
    print (file)
    copyfile(file,"./WAVFiles/"+file.split('/')[-1])
