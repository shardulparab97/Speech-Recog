# This python script preforms an MFCC analysis of every .wav file in a specified directory and saves the filterbank
# energies for each file in a new directory (as a .csv).
#
# Please note: running this script multiple times will overwrite earlier analyses
#
# Requires python_speech_features. Documentation: https://github.com/jameslyons/python_speech_features)
# Requires scipy. Scipy documentation: https://www.scipy.org/install.html
#
# Script written by Rachael Tatman (rachael.tatman@gmail.com), supported by National Science Foundation grant DGE-1256082

# import things we're going to need
import json
import features
#from features import logfbank
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import scipy.io.wavfile as wav
import numpy
import os

# directory where we your .wav files are
directoryName = "WAVFiles" # put your own directory here
# directory to put our results in, you can change the name if you like
resultsDirectory = "mfcc"
# make a new folder in this directory to save our results in
if not os.path.exists(resultsDirectory):
    os.makedirs(resultsDirectory)

# get MFCCs for every .wav file in our specified directory
for filename in os.listdir(directoryName):
    if filename.endswith('.WAV'): # only get MFCCs from .wavs
        # read in our file
        (rate,sig) = wav.read(directoryName + "/" +filename)

        # get mfcc
        mfcc_feat = mfcc(sig,rate)

        # get filterbank energies
        fbank_feat = logfbank(sig,rate)
        print("Shape of ", filename,"is:",numpy.shape(fbank_feat))

        # create a file to save our results in
        outputFile = resultsDirectory + "/" + os.path.splitext(filename)[0] + ".mfcc"
        #with open(outputFile,'wb') as f:
        #    numpy.savetxt(f,fbank_feat.flatten())
        with open(outputFile, 'w') as f:
         json.dump(fbank_feat.tolist(), f, allow_nan = True)
        #outputFile=resultsDirectory+"/yolo.mfcc"
        #file = open(outputFile, 'w+')
        #(fbank_feat)# make file/over write existing file
        #numpy.savetxt(file, fbank_feat) #save MFCCs as .csv
        #ile.close() # close file

'''Bash code

for file in ./WAVFiles/*
do
    sox "$file" -e signed-integer "$file" >> results.out
done'''
