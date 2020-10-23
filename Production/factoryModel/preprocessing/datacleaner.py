'''
Script for preprocessing data for Machine Translation application
This is the class for removing the punctuations from sentences and also converting it to lower cases
'''

import string
from numpy import array
from unicodedata import normalize

class cleanData:
	def __init__(self):
		# Creating the constructor for removing punctuations and lowering the text
		pass
		
	# Creating the function for removing the punctuations and converting to lowercase
	def preprocess(self,lines):
		cleanArray = list()
		for docs in lines:
			cleanDocs = list()
			for line in docs:
				# Normalising unicode characters
				line = normalize('NFD', line).encode('ascii', 'ignore')
				line = line.decode('UTF-8')
				# Tokenize on white space
				line = line.split()
				# Removing punctuations from each token
				line = [word.translate(str.maketrans('', '', string.punctuation)) for word in line]
				# convert to lower case
				line = [word.lower() for word in line]
				# Remove tokens with numbers in them
				line = [word for word in line if word.isalpha()]
				# Store as string
				cleanDocs.append(' '.join(line))
			cleanArray.append(cleanDocs)
		return array(cleanArray)
	