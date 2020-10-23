'''
Script for preprocessing of text for Machine Translation
This is the class for splitting the text into sentences
'''

import string
from numpy import array

class SentenceSplit:
	def __init__(self,nrecords):
		# Creating the constructor for splitting the sentences
		# nrecords is the parameter which defines how many records you want to take from the data set
		self.nrecords = nrecords
		
	# Creating the new function for splitting the text
	def preprocess(self,text):
		sen = text.strip().split('\n')
		sen = [i.split('\t') for i in sen]
		# Saving into an array
		sen = array(sen)
		# Return only the first two columns as the third column is metadata. Also select the number of rows required
		return sen[:self.nrecords,:2]
	
	
	
	
















