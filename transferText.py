import os
import shutil
import glob
from shutil import copyfile
import numpy as np

count=0
for subdir, dirs, files in os.walk('./TIMIT'):
        for file in files:
            fullFilename = os.path.join(subdir, file)
            filenameNoSuffix =  os.path.splitext(fullFilename)[0]
            if file.endswith('.WAV'):
                labelFilename = filenameNoSuffix + '.TXT'
                text = []
                with open(labelFilename,'r') as f:
                    content=f.read()
                    char_count=0
                    content_string=""
                    for char in content:
                        if char_count>7:
                            content_string+=char
                        char_count+=1
                    print(content_string)
                    content_string=np.array(content_string)
                    count+=1
                    print("File Number: " ,count)
                    labelFilename = './Text_Targets/' + filenameNoSuffix.split('/')[-2]+'-'+filenameNoSuffix.split('/')[-1]+'.npy'
                    print(labelFilename)
                    np.save(labelFilename,content_string)

