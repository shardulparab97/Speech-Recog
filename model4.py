import tensorflow as tf
import numpy as np
from numpy import matrix
from tensorflow.python.ops import ctc_ops as ctc
init=tf.global_variables_initializer()
sess=tf.Session()
import common
#indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
#values = np.array([1.0, 2.0], dtype=np.float32)
#shape = np.array([7, 9, 2], dtype=np.int64)
sess.run(init)
num_features=13
num_classes=63 #still decide is it 39?? since we have made it into 39 phonemes accordingly

num_epochs = 10000
num_hidden = 100
num_layers = 1
batch_size = 1
#print(sess.run(targets,feed_dict={targets:(indices,values,shape)}))
OUTPUT_SHAPE=(100,62)
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

    indices = np.asarray(indices, dtype=np.int64)
    indices[:,[0, 1]] = indices[:,[1, 0]]
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

    #logits3d = tf.stack(logits)
    #loss = tf.reduce_mean(ctc.ctc_loss(targets, logits3d, seq_len))
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
    for curr_epoch in range(num_epochs):
        train_cost=train_ler=0
        count_files=0
        for file in htmlfiles:
            fullFilename = os.path.join(file)
            #print(fullFilename)
            filenameNoSuffix =  os.path.splitext(fullFilename)[0]
            count=0
            count_files+=1
            print("File_number= ",count_files)
            print (filenameNoSuffix.split('/')[-1])
            for file1 in htmlfiles1:
                fullFilename1 = os.path.join(file1)
                filenameNoSuffix1 =  os.path.splitext(fullFilename1)[0]
                if(filenameNoSuffix1.split('/')[-1]==filenameNoSuffix.split('/')[-1]):
                    #print(filenameNoSuffix1.split('/')[-1])
                    count+=1
                    #printing the file name
                    # print (filenameNoSuffix1.split('/')[-1])
                    #print("*************Input data**************")
                    input_val = np.load("./mfccnpyFiles/" +filenameNoSuffix.split('/')[-1]+'.npy')
                    #print (np.shape(input_val))
                    #print("Output phoneme val:*****************")
                    output_val=np.load("./phonemeLabels/" +filenameNoSuffix.split('/')[-1]+'.npy')
                    #print("Sequence length is:",np.shape(output_val)[0])
                    #count2=count2+1
                    #print (count2)
                    #print (np.shape(input_val))
                    train_seq_len=[92]
                    input_val=np.reshape(input_val,[1,92,13])

                    sparse_targets_indices,sparse_targets_values,sparse_targets_shapes = sparse_tuple_from(output_val)
                    #print("Required stuf:",max(sparse_targets_indices,sparse_targets_indices[:1]==1,2))
                    #print ("*******************",np.shape(sparse_targets_indices))
                    #print (sparse_targets_indices,sparse_targets_values,sparse_targets_shapes)
                    #print ("The value here is:",session.run(targets,feed_dict={targets: tf.SparseTensorValue(sparse_targets_indices,sparse_targets_values,sparse_targets_shapes)}))
                    #print ("*******SHAPE OF SPARSE INDICES=***********",np.shape(sparse_targets_indices))

                    #print ("*******SHAPE OF SPARSE VALS=***********",np.shape(sparse_targets_values))
                    #print ("*******SHAPE OF SPARSE SHAPE=***********",np.shape(sparse_targets_shapes))
                    #sparse_targets_indices=[[0,0], [0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8]]
                    #sparse_targets_values=[1.0, 2.0,3,4,5,6,7,8,9]
                    #sparse_targets_shapes=[9,1]
                    sp=tf.SparseTensor(indices=sparse_targets_indices,values=sparse_targets_values,dense_shape=sparse_targets_shapes)
                    #sess.run(sp.eval())
                    batch_cost,_ =  session.run([cost,optimizer], feed_dict={inputs: input_val, targets: tf.SparseTensorValue(sparse_targets_indices,sparse_targets_values,sparse_targets_shapes) , seq_len: train_seq_len})
                    train_cost += batch_cost * batch_size
                    train_ler += session.run(ler,feed_dict={inputs: input_val, targets: tf.SparseTensorValue(sparse_targets_indices,sparse_targets_values,sparse_targets_shapes), seq_len: train_seq_len}) * batch_size

                        #begin with decoding
                    d = session.run(decoded[0], feed_dict={inputs: input_val, targets: tf.SparseTensorValue(sparse_targets_indices,sparse_targets_values,sparse_targets_shapes), seq_len: train_seq_len})
                    print ("Batch cost:",batch_cost)
                    if(count>=1):
                        break
        print("TRAIN COST:",train_cost)
        print ("TRAIN LER:",train_ler)

new_time = time.time()
print ("Total time taken is :",(new_time-last_time))
