We have 3 important files:
1)transfer.py:VERY IMPORTANT for transferring all the files of one file type in the TIMIT to one folder .
Example storing all the .WAV files in one folder. Here it is WAVFiles Folder.

2)Run the bash script to convert it into a format which can be understood by python_speech_feaures

for file in ./WAVFiles/*
do
    sox "$file" -e signed-integer "$file" >> results.out
done

3)#Now run wav2mfcc.py in order to convert the .WAV files in the /WAVFiles folder and convert it to mfcc and store it in the
  #"mfcc" folder.
    
Instead run wav2mfccnpy.py which stores the .WAV files as mfcc numpy arrays which can be easily saved and restored as numpy array.
Now run the file and the mfcc.npy files are stored in the "mfccnpyFiles" folder
 Each of the npy files contains a numpy array of size 9X13(9 frames X 13 features)
4)phonemeFiles has all the phoneme file together from TIMIT.

5)phonemeLabels is the folder which has the phonemes from the phoneme files mapped according to the phoneme dictionary.(6300 files)

6)Use "phoneme.py" has the functions to do step 4 and 5.


WORK TO DO LATER ON:
- Use the transfer code as a standard fuction to do transferring of any file type passed as an argument.
- Randomizing selection of files
- Working with more than one batch
- Compare decoding of both repos
- [ ] (priority)CTC beam decoder

OTHER FILES:
1)Test1.py shows how to use the mfccnpy and phonemeLabel folders and take their files and choose files with the same name, take
 out the numpy arrays which can be then used as input and output

STATISTICS:
Total time to read data from the mfccnpy files+ phonemelabel files is:
    Without buffer(first read) : 33s
    With next try: 6s
    
CHALLENGES:
The number of phonemes in every utterance are different.

IMPORTANT FUNCTIONS:
FROM zzw922cn:
output_to_sequence in utils/utils.py
utils/calPer.py - very important for sparse_tensor_to_seq_list

INITIAL SETUP FOR THE NETWORK:
The network is defined as:
- One LSTM layer `rnn.LSTMCell` with 100 units, completed by a softmax.
- Batch size of 1.
- Momentum Optimizer with learning rate of 0.005 and momentum of 0.9.

OBSERVATIONS:
(1) is zzw922n,(2) is philipperemy
- 1 used 39 phonemes but has only 30 classes
- has straightforward used character wise classification

QUESTIONS:
- Usage of sparse tensors for ctc loss
- what is ctc_greedy_decoder for accuracy??
