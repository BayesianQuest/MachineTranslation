'''
This script lists down all the helper functions which are required for processing raw data
'''

from pickle import load
from numpy import argmax
from pickle import dump
from tensorflow.keras.preprocessing.sequence import pad_sequences
from numpy import array
from unicodedata import normalize
import string

# Function to Save data to pickle form
def save_clean_data(data,filename):
    dump(data,open(filename,'wb'))
    print('Saved: %s' % filename)

# Function to load pickle data from disk
def load_files(filename):
    return load(open(filename,'rb'))

# Function to clean the input data
def cleanInput(lines):
    cleanSent = []
    cleanDocs = list()
    for docs in lines[0].split():
        line = normalize('NFD', docs).encode('ascii', 'ignore')
        line = line.decode('UTF-8')
        line = [line.translate(str.maketrans('', '', string.punctuation))]
        line = line[0].lower()
        cleanDocs.append(line)
    cleanSent.append(' '.join(cleanDocs))
    return array(cleanSent)

# Function to convert sentences to sequences of integers
def encode_sequences(tokenizer,length,lines):
    # Sequences as integers
    X = tokenizer.texts_to_sequences(lines)
    # Padding the sentences with 0
    X = pad_sequences(X,maxlen=length,padding='post')
    return X

# Generate target sentence given source sequence
def Convertsequence(tokenizer,source):
    target = list()
    reverse_eng = tokenizer.index_word
    for i in source:
        if i == 0:
            continue
        target.append(reverse_eng[int(i)])
    return ' '.join(target)

# Function to generate predictions from source data
def generatePredictions(model,tokenizer,data):
    prediction = model.predict(data,verbose=0)
    AllPreds = []
    for i in range(len(prediction)):
        predIndex = [argmax(prediction[i, :, :], axis=-1)][0]
        target = Convertsequence(tokenizer,predIndex)
        AllPreds.append(target)
    return AllPreds



