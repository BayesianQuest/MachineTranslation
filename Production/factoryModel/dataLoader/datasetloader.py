'''
Factory Model for Machine translation preprocessing.
This is the script for loading the data and preprocessing data
'''

import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array

# Creating the class to load data and then do the preprocessing as sequence of steps

class textLoader:
	def __init__(self , preprocessors = None):
		# This init method is to store the text preprocessing pipeline
		self.preprocessors = preprocessors
		# Initializing the preprocessors as an empty list of the preprocessors are None
		if self.preprocessors is None:
			self.preprocessors = []

	def loadDoc(self,filepath):
		# This is the function to read the file from the path provided
		# Open the file
		file = open(filepath,mode = 'rt',encoding = 'utf-8')
		# Reading the text
		text = file.read()
		#Once the file is read, applying the preprocessing steps one by one
		if self.preprocessors is not None:
			# Looping over all the preprocessing steps and applying them on the text data
			for p in self.preprocessors:
				text = p.preprocess(text)
				
		# Closing the file
		file.close()
				
		# Returning the text after all the preprocessing
		return text
		