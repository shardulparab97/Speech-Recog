import io
import collections
import tensorflow as tf
import time
import numpy as np
import os
import shutil
from numpy import matrix
from tensorflow.python.ops import ctc_ops as ctc
import argparse
import glob
import pickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--action")
parser.add_argument("--model_no",default="0")
parser.parse_args()
args = parser.parse_args()
#print(args.act)

def read_data():
    text_files = [os.path.join(root, name) for root, dirs, files in os.walk('Text_Targets') for name in files
                 if name.endswith((".npy"))]
    full_text=""
    for file in text_files:
        count=0
        text_No_Suffix=os.path.splitext(file)[0]
        text_File_Name=text_No_Suffix.split('/')[-1]
        content=str(np.load("./Text_Targets/"+text_File_Name+'.npy'))
        content = io.StringIO(content).readlines()
        content = (((((content[0].replace("\n","")).replace(".","")).replace("?", "")).replace('"','')).replace("!","")).replace(",","")
        content = (((content.replace(":","")).replace("'","")).replace("-","")).replace(";","")
        full_text+=(' '+content)
    return full_text.lower()

def build_dataset(words):
    list_val=list(set(words))
    dictionary = dict()
    for word in list_val:
        dictionary[word] = len(dictionary) #Basically assigns a number to each word
    dictionary={'e': 0, 'u': 1, 'h': 24, 'v': 14, 'w': 15, 'j': 2, 'n': 16, 'y': 3, 'b': 4, 't': 18, 'a': 19, 'm': 5, 'x': 23, 'p': 20, 'c': 21, 'g': 6, ' ': 22, 'z': 7, 'r': 8, 'f': 9, 'k': 10, 's': 11, 'q': 25, 'd': 12, 'i': 26, 'l': 17, 'o': 13}
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary


data=read_data()
dictionary,reverse_dictionary=build_dataset(data)
print(dictionary)
#dictionary={'e': 0, 'u': 1, 'h': 24, 'v': 14, 'w': 15, 'j': 2, 'n': 16, 'y': 3, 'b': 4, 't': 18, 'a': 19, 'm': 5, 'x': 23, 'p': 20, 'c': 21, 'g': 6, ' ': 22, 'z': 7, 'r': 8, 'f': 9, 'k': 10, 's': 11, 'q': 25, 'd': 12, 'i': 26, 'l': 17, 'o': 13}
print(len(dictionary))

n_classes=28
n_features=13
#Hyperparameters
num_layers=2
n_hidden=150
batch_size=1
n_epochs=1000

#Target log path
logs_path = '/tmp/tensorflow/timit_speech_recognition'

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(matrix(seq)), range(len(matrix(seq)))))
        values.extend([seq])

    indices = np.asarray(indices, dtype=np.int32)
    indices[:,[0, 1]] = indices[:,[1, 0]]
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int32)

    return indices, values, shape


inputs = tf.placeholder(tf.float32,[None,None,n_features])
target_idx = tf.placeholder(tf.int64)
target_vals = tf.placeholder(tf.int32)
target_shape = tf.placeholder(tf.int64)
targets = tf.SparseTensor(target_idx, target_vals, target_shape)
seq_len = tf.placeholder(tf.int32)

# RNN output node weights and biases
weights = {
        'out': tf.Variable(tf.random_normal([n_hidden,n_classes]),dtype=tf.float32) # Weights_shape = hidden_units X vocab_size
    }
biases = {
        'out': tf.Variable(tf.random_normal([n_classes]),tf.float32)
        }

def LSTM_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, reuse=tf.get_variable_scope().reuse)
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.4)

def RNN_Model(inputs,seq_len,weights,biases):
    rnn_cell=LSTM_cell()
    #rnn_cell = tf.contrib.rnn.MultiRNNCell(LSTM_cell() for i in range(num_layers))
    outputs,_ = tf.nn.dynamic_rnn(rnn_cell,inputs,seq_len,dtype=tf.float32)

    outputs = tf.reshape(outputs,[-1,n_hidden])
    #print(outputs)
    logits = tf.matmul(outputs,weights['out']) + biases['out']

    with tf.name_scope('Weights'):
        variable_summaries(weights['out'])

    with tf.name_scope('Biases'):
        variable_summaries(biases['out'])

    with tf.name_scope('Activations'):
        tf.summary.histogram('Activations',logits)

    logits = tf.reshape(logits,[batch_size,-1,n_classes])
    logits = tf.transpose(logits,(1,0,2))
    return logits

logits=RNN_Model(inputs,seq_len,weights,biases)
loss = ctc.ctc_loss(targets,logits,seq_len)
with tf.name_scope("CTC_Loss"):
    cost = tf.reduce_mean(loss)
    tf.summary.scalar('CTC_Loss',cost)

optimizer = tf.train.MomentumOptimizer(learning_rate=0.005, momentum=0.9).minimize(cost)
decoded, log_prob = ctc.ctc_greedy_decoder(logits, seq_len)

label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                              targets))
tf.summary.scalar('Label Error Rate',label_error_rate)
#computes the Levenshtein distance between sequences.
#This operation takes variable-length sequences (hypothesis and truth), each provided as a SparseTensor, and computes the Levenshtein distance.
# You can normalize the edit distance by length of truth by setting normalize to true.

def convert_to_sequence(values):
    result=""
    for value in values:
        result+=reverse_dictionary[value]
    return result


merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(logs_path)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
saver = tf.train.Saver(tf.global_variables())
init=tf.global_variables_initializer()
starting_epoch=0
def train():
    with tf.Session(config=config) as sess:
            if(args.model_no=="0"):
                print("RUNNING FROM BEGINNING FROM BEGINNING")
                starting_epoch=0
            else:
                starting_epoch=int(args.model_no)

            list_of_files = glob.glob('./saver/*') # * means all if need specific format then *.csv
            latest_file_no=0
            if(len(list_of_files)!=0):
                latest_file = max(list_of_files, key=os.path.getctime)
                latest_file_no=latest_file.split(".meta")[0].split("-")[-1].split(".ckpt")[0]
            if(starting_epoch>int(latest_file_no)):
                print ("Starting epoch value greater than available epochs. Starting from last epoch run")
                starting_epoch=int(latest_file_no)

            if(args.action=='r'):
                sess.run(init)
                save_path="./saver/model-"+str(starting_epoch)+".ckpt"
                saver.restore(sess,save_path)
                print("Restored")
                starting_epoch+=1
                print(sess.run(weights['out']))
            elif(args.action=='s'):
                sess.run(init)
            writer.add_graph(sess.graph)
            mfcc_files = [os.path.join(root, name) for root, dirs, files in os.walk('mfccnpyFiles') for name in files
             if name.endswith((".npy"))]
            text_targets=[os.path.join(root, name) for root, dirs, files in os.walk('Text_Targets') for name in files
             if name.endswith((".npy"))]
            for curr_epoch in range(starting_epoch,n_epochs):
                train_cost=train_ler=0
                count_files=0
                for file in mfcc_files:
                    count=0
                    mfcc_No_Suffix=os.path.splitext(file)[0]
                    mfcc_File_Name=mfcc_No_Suffix.split('/')[-1]
                    for text in text_targets:
                        text_No_Suffix=os.path.splitext(text)[0]
                        text_File_Name=text_No_Suffix.split('/')[-1]
                        if(mfcc_File_Name==text_File_Name):
                            count+=1
                            train_inputs=np.load("./mfccnpyFiles/"+mfcc_File_Name+'.npy')
                            train_inputs = np.asarray(train_inputs[np.newaxis, :])
                            train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
                            train_seq_len = [train_inputs.shape[1]]
                            #print(train_seq_len)

                            content=str(np.load("./Text_Targets/"+text_File_Name+'.npy'))
                            content = io.StringIO(content).read()
                            content = (((((content.replace("\n","")).replace(".","")).replace("?", "")).replace('"','')).replace("!","")).replace(",","")
                            content = (((content.replace(":","")).replace("'","")).replace("-","")).replace(";","")
                            content=list(content.lower())
                            #print(content)
                            train_targets=[]
                            for word in content:
                                train_targets.append(dictionary[word])
                            train_targets=np.array(train_targets)

                            target_index,target_values,target_shapes = sparse_tuple_from(train_targets)
                            feed = {inputs: train_inputs,target_idx:target_index,target_vals:target_values,target_shape:target_shapes,seq_len: train_seq_len}
                            summary,batch_cost, _ = sess.run([merged, cost, optimizer], feed)
                            #writer.add_summary(summary, count_files)

                            train_cost += batch_cost * batch_size
                            ler_cost=0
                            print('Truth:\n' + convert_to_sequence(train_targets))
                            print('Output:\n' + convert_to_sequence(sess.run(decoded[0].values,feed_dict=feed)))
                            #print("Decoded Values: " ,sess.run(decoded[0].values,feed_dict=feed))
                            train_ler += sess.run(label_error_rate, feed_dict=feed) * batch_size
                            count_files+=1
                            print("Batch Cost {0} after {1} file ,in epoch {2}\n".format(batch_cost,count_files,curr_epoch))
                            print("LER {0} after {1} file ,in epoch {2}\n".format(train_ler,count_files,curr_epoch))
                            if(count_files==6300):
                                train_cost_per_epoch=(train_cost/6300)
                                ler_cost=train_ler/6300
                                tf.summary.scalar('Train_Cost',train_cost_per_epoch)
                                tf.summary.scalar('Train LER',ler_cost)
                                writer.add_summary(summary, curr_epoch)


                        if(count>=1):
                            break
                save_path="./saver/model-"+str(curr_epoch)+".ckpt"
                print(" Train Cost: {0} in Epoch {1} ".format((train_cost),curr_epoch))
                print(" Training LER: {0} in Epoch {1} ".format((ler_cost),curr_epoch))
                saver.save(sess,save_path)
                print("Model saved ")



train()
