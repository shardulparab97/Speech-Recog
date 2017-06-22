import tensorflow as tf
import sys
import os
sys.path.append('../')
sys.dont_write_bytecode = True
from glob import glob
import numpy as np
import tensorflow as tf
import math

def check_path_exists(path):
    """ check a path exists or not
    """
    if isinstance(path, list):
        for p in path:
            if not os.path.exists(p):
                os.makedirs(p)
    else:
        if not os.path.exists(path):
            os.makrdirs(path)

def load_batched_data(mfccPath, labelPath, batchSize, mode, level):
    '''returns 3-element tuple: batched data (list), maxTimeLength (int), and
       total number of samples (int)'''
    return data_lists_to_batches([np.load(os.path.join(mfccPath, fn)) for fn in os.listdir(mfccPath)],
                                 [np.load(os.path.join(labelPath, fn)) for fn in os.listdir(labelPath)],
                                 batchSize, level) + \
            (len(os.listdir(mfccPath)),)

def list_dirs(mfcc_dir, label_dir):
    mfcc_dirs = glob(mfcc_dir)
    label_dirs = glob(label_dir)
    for mfcc,label in zip(mfcc_dirs,label_dirs):
        yield (mfcc,label)

def data_lists_to_batches(inputList, targetList, batchSize, level):
    ''' padding the input list to a same dimension, integrate all data into batchInputs
    '''
    assert len(inputList) == len(targetList)
    # dimensions of inputList:batch*39*time_length

    nFeatures = inputList[0].shape[0]
    maxLength = 0
    for inp in inputList:
	    # find the max time_length
        maxLength = max(maxLength, inp.shape[1])

    # randIxs is the shuffled index from range(0,len(inputList))
    randIxs = np.random.permutation(len(inputList))
    start, end = (0, batchSize)
    dataBatches = []

    while end <= len(inputList):
	    # batchSeqLengths store the time-length of each sample in a mini-batch
        batchSeqLengths = np.zeros(batchSize)

  	    # randIxs is the shuffled index of input list
        for batchI, origI in enumerate(randIxs[start:end]):
            batchSeqLengths[batchI] = inputList[origI].shape[-1]

        batchInputs = np.zeros((maxLength, batchSize, nFeatures))
        batchTargetList = []
        for batchI, origI in enumerate(randIxs[start:end]):
	        # padSecs is the length of padding
            padSecs = maxLength - inputList[origI].shape[1]
	        # numpy.pad pad the inputList[origI] with zeos at the tail
            batchInputs[:,batchI,:] = np.pad(inputList[origI].T, ((0,padSecs),(0,0)), 'constant', constant_values=0)
	        # target label
            batchTargetList.append(targetList[origI])
        dataBatches.append((batchInputs, list_to_sparse_tensor(batchTargetList, level), batchSeqLengths))
        start += batchSize
        end += batchSize
    return (dataBatches, maxLength)

def list_to_sparse_tensor(targetList, level):
    ''' turn 2-D List to SparseTensor
    '''
    indices = [] #index
    vals = [] #value
    assert level == 'phn' or level == 'cha', 'type must be phoneme or character, seq2seq will be supported in future'
    phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h',\
       'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl',\
       'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng',\
       'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#',\
       'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k',\
       'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow',\
       'oy', 'p', 'pau', 'pcl', 'q', 'r', 's',\
       'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux',\
       'v', 'w', 'y', 'z', 'zh']

    mapping = {'ah': 'ax', 'ax-h': 'ax', 'ux': 'uw', 'aa': 'ao', 'ih': 'ix', \
               'axr': 'er', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n',\
               'eng': 'ng', 'sh': 'zh', 'hv': 'hh', 'bcl': 'h#', 'pcl': 'h#',\
               'dcl': 'h#', 'tcl': 'h#', 'gcl': 'h#', 'kcl': 'h#',\
               'q': 'h#', 'epi': 'h#', 'pau': 'h#'}

    group_phn = ['ae', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', \
                 'er', 'ey', 'f', 'g', 'h#', 'hh', 'ix', 'iy', 'jh', 'k', 'l', \
                 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 't', 'th', 'uh', 'uw',\
                 'v', 'w', 'y', 'z', 'zh']


    mapping = {'ah': 'ax', 'ax-h': 'ax', 'ux': 'uw', 'aa': 'ao', 'ih': 'ix', \
               'axr': 'er', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n',\
               'eng': 'ng', 'sh': 'zh', 'hv': 'hh', 'bcl': 'h#', 'pcl': 'h#',\
               'dcl': 'h#', 'tcl': 'h#', 'gcl': 'h#', 'kcl': 'h#',\
               'q': 'h#', 'epi': 'h#', 'pau': 'h#'}

    group_phn = ['ae', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', \
                 'er', 'ey', 'f', 'g', 'h#', 'hh', 'ix', 'iy', 'jh', 'k', 'l', \
                 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 't', 'th', 'uh', 'uw',\
                 'v', 'w', 'y', 'z', 'zh']

    if level == 'cha':
        for tI, target in enumerate(targetList):
            for seqI, val in enumerate(target):
                indices.append([tI, seqI])
                vals.append(val)
        shape = [len(targetList), np.asarray(indices).max(axis=0)[1]+1] #shape
        return (np.array(indices), np.array(vals), np.array(shape))

    elif level == 'phn':
        '''
        for phn level, we should collapse 61 labels into 39 labels before scoring

        Reference:
          Heterogeneous Acoustic Measurements and Multiple Classifiers for Speech Recognition(1986),
            Andrew K. Halberstadt, https://groups.csail.mit.edu/sls/publications/1998/phdthesis-drew.pdf
        '''
        for tI, target in enumerate(targetList):
            for seqI, val in enumerate(target):
                if val < len(phn) and (phn[val] in mapping.keys()):
                    val = group_phn.index(mapping[phn[val]])
                indices.append([tI, seqI])
                vals.append(val)
        shape = [len(targetList), np.asarray(indices).max(0)[1]+1] #shape
        return (np.array(indices), np.array(vals), np.array(shape))

    else:
        ##support seq2seq in future here
        raise ValueError('Invalid level: %s'%str(level))

def load_data(mode, type):
    if mode == 'train':
        return load_batched_data(train_mfcc_dir, train_label_dir, batch_size, mode, type)
    elif mode == 'test':
        return load_batched_data(test_mfcc_dir, test_label_dir, batch_size, mode, type)
    else:
        raise TypeError('mode should be train or test.')
datadir="/home/shardulparab97/Desktop/LiveWeaver/tensorflowpractice/speechRecognition/data/lisa/data/timit/raw/TIMIT/TRAIN/DR1"
level="phn"
train_mfcc_dir = os.path.join(datadir, level, 'train', 'mfcc')
train_label_dir = os.path.join(datadir, level, 'train', 'label')
test_mfcc_dir = os.path.join(datadir, level, 'test', 'mfcc')
test_label_dir = os.path.join(datadir, level, 'test', 'label')
batch_size=100

batchedData, maxTimeSteps, totalN = load_data(mode='train', type=level)
