import os
import shutil
import glob
from shutil import copyfile
import numpy as np


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
def transfer_phonemes():
    htmlfiles = [os.path.join(root, name)
             for root, dirs, files in os.walk('TIMIT')
             for name in files
             if name.endswith((".PHN"))]


    for file in htmlfiles:
        copyfile(file,"./phonemeFiles/"+file.split('/')[-1])

phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

def function2(rootdir,label_dir,save_directory):#This function makes a dictionary of the phonemes in the file mapping with phn
    count = 0
    for subdir, dirs, files in os.walk(rootdir):
         for file in files:
            fullFilename = os.path.join(subdir, file)
            filenameNoSuffix =  os.path.splitext(fullFilename)[0]
            #gets the entire path from timit to end
            #eg:TIMIT/TRAIN/DR8/MMWS0/SX78
            #print (filenameNoSuffix)
            if file.endswith('.WAV'):
                labelFilename = filenameNoSuffix + '.PHN'
                phenome = []
                with open(labelFilename,'r') as f:
                    phenome.append(len(phn))
                    for line in f.read().splitlines():
                            s=line.split(' ')[2]
                            p_index = phn.index(s)
                            phenome.append(p_index)

                    phenome.append(len(phn)+1) # <end token>
                    print(phenome)
                    phenome = np.array(phenome)
                    count+=1
                    print ("File index:",count)
                    #important Line
                    labelFilename = label_dir + filenameNoSuffix.split('/')[-2]+'-'+filenameNoSuffix.split('/')[-1]+'.npy'
                    print(labelFilename)
                    np.save(labelFilename,phenome)

function2('TIMIT/','./phonemeLabels/','sjshsj')
#Use the below code for creating label npy files that is
'''import numpy as np
data = np.load('./phonemeLabels/FADG0-SA1.npy')

print(data)'''
