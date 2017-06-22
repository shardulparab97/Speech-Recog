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

#boilerplate code for transferring files with a particular extension
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

'''Bash code

for file in ./WAVFiles/*
do
    sox "$file" -e signed-integer "$file" >> results.out
done'''
(rate,sig) = wav.read("SX37.WAV")
mfcc_feat = mfcc(sig,rate)
#fbank_feat = logfbank(sig,rate)

print(numpy.shape(mfcc_feat))
