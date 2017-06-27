import tensorflow as tf
import numpy as np
from numpy import matrix
from tensorflow.python.ops import ctc_ops as ctc
init=tf.global_variables_initializer()
sess=tf.Session()
#indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
#values = np.array([1.0, 2.0], dtype=np.float32)
#shape = np.array([7, 9, 2], dtype=np.int64)
sess.run(init)
num_features=13
num_classes=30 #still decide is it 39?? since we have made it into 39 phonemes accordingly

num_epochs = 10000
num_hidden = 100
num_layers = 1
batch_size = 1
#print(sess.run(targets,feed_dict={targets:(indices,values,shape)}))

import os
import shutil
import glob
from shutil import copyfile
import numpy as np
import time

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []
    #print ("SEQUENCES IN FUNCTION:",sequences)

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(matrix(seq)), range(len(matrix(seq)))))
        values.extend([seq])

    indices = np.asarray(indices, dtype=np.int32)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int32)

    return indices, values, shape


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
graph=tf.Graph()
with graph.as_default():
    inputs=tf.placeholder(tf.float32, [None,None,num_features],name="INPUTS")

    targets = tf.sparse_placeholder(tf.int32,name="targets")


    #targets = tf.placeholder(tf.int32,[None],name="targets")
    seq_len=tf.placeholder(tf.int32,[None],name="sequence_length")

    cell=tf.contrib.rnn.BasicLSTMCell(num_hidden)
    stack = tf.contrib.rnn.MultiRNNCell([cell]*num_layers)
    #see what seq_len will do??
    outputs,_ = tf.nn.dynamic_rnn(stack,inputs,seq_len, dtype=tf.float32)

    #nice stuff here!
    shape=tf.shape(inputs)
    batch_s, max_time_steps = shape[0],shape[1]

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_hidden])

    W = tf.Variable(tf.truncated_normal([num_hidden,num_classes],stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[num_classes]))

    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    logits = tf.transpose(logits, (1, 0, 2))

    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    #(labels, inputs, sequences) #returns loss tensor of size batch
    cost = tf.reduce_mean(loss)

    #optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.005, momentum=0.9).minimize(cost)

    #This is a bit slower, instead you can do this:
    #acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),targets))
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

    #computes the Levenshtein distance between sequences.
    #This operation takes variable-length sequences (hypothesis and truth), each provided as a SparseTensor, and computes the Levenshtein distance.
    # You can normalize the edit distance by length of truth by setting normalize to true.
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),targets))


    init=tf.global_variables_initializer()
with tf.Session(graph=graph) as session:
    session.run(init)
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
                print("Sequence length is:",np.shape(output_val)[0])
                #count2=count2+1
                #print (count2)
                train_seq_len=np.shape(input_val[0])
                input_val=np.reshape(input_val,[-1,9,13])

                sparse_targets_indices,sparse_targets_values,sparse_targets_shapes = sparse_tuple_from(output_val)
                print (sparse_targets_indices,sparse_targets_values,sparse_targets_shapes)
                print ("The value here is:",session.run(targets,feed_dict={targets: tf.SparseTensorValue(sparse_targets_indices,sparse_targets_values,sparse_targets_shapes)}))
                batch_cost,_ =  session.run([cost,optimizer], feed_dict={inputs: input_val, targets: tf.SparseTensorValue(sparse_targets_indices,sparse_targets_values,sparse_targets_shapes) , seq_len: train_seq_len})
                if(count>=1):
                    break


new_time = time.time()
print ("Total time taken is :",(new_time-last_time))
