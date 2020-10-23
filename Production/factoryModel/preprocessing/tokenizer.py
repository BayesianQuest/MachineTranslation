'''
This class has methods for tokenizing the text and preparing train and test sets
'''

import string
import numpy as np
from numpy import array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


class TrainMaker:
	def __init__(self):
		# Creating the constructor for creating the tokenizers
		pass
	
	# Creating an internal function for tokenizing the text	
	def tokenMaker(self,text):
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(text)
		return tokenizer
		
	# Creating an internal function for encoding and padding sequences
	
	def sequenceMaker(self,tokenizer,stdlen,text):
		# Encoding sequences as integers
		seq = tokenizer.texts_to_sequences(text)
		# Padding the sequences with respect standard length
		seq = pad_sequences(seq,maxlen=stdlen,padding = 'post')
		return seq
		
	# Creating another function to find the maximum length of the sequences	
	def qntLength(self,lines):
		doc_len = []
		# Getting the length of all the language sentences
		[doc_len.append(len(line.split())) for line in lines]
		return np.quantile(doc_len, .975)
		
	# Creating the function for creating tokenizers and also creating the train and test sets from the given text
	def preprocess(self,docArray):
		# Creating tokenizer forEnglish sentences
		eng_tokenizer = self.tokenMaker(docArray[:,0])
		# Finding the vocabulary size of the tokenizer
		eng_vocab_size = len(eng_tokenizer.word_index) + 1
		# Creating tokenizer for German sentences
		deu_tokenizer = self.tokenMaker(docArray[:,1])
		# Finding the vocabulary size of the tokenizer
		deu_vocab_size = len(deu_tokenizer.word_index) + 1
		# Finding the maximum length of English and German sequences
		eng_length = self.qntLength(docArray[:,0])
		ger_length = self.qntLength(docArray[:,1])
		# Splitting the train and test set
		train,test = train_test_split(docArray,test_size = 0.1,random_state = 123)
		# Calling the sequence maker function to create sequences of both train and test sets
		# Training data
		trainX = self.sequenceMaker(deu_tokenizer,int(ger_length),train[:,1])
		trainY = self.sequenceMaker(eng_tokenizer,int(eng_length),train[:,0])
		# Validation data
		testX = self.sequenceMaker(deu_tokenizer,int(ger_length),test[:,1])
		testY = self.sequenceMaker(eng_tokenizer,int(eng_length),test[:,0])
		return eng_tokenizer,eng_vocab_size,deu_tokenizer,deu_vocab_size,docArray,trainX,trainY,testX,testY,eng_length,ger_length