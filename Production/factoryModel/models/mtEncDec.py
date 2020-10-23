'''
This is the script and template for different models.
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed

class ModelBuilding:
	@staticmethod
	def EncDecbuild(in_vocab,out_vocab, in_timesteps,out_timesteps,units):
		# Initializing the model with Sequential class
		model = Sequential()
		# Initiating the embedding layer for the text
		model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
		# Adding the first LSTM layer
		model.add(LSTM(units))
		# Using the RepeatVector to map the input sequence length to output sequence length
		model.add(RepeatVector(out_timesteps))
		# Adding the second layer of LSTM 
		model.add(LSTM(units, return_sequences=True))
		# Adding the fully connected layer with a softmax layer for getting the probability
		model.add(TimeDistributed(Dense(out_vocab, activation='softmax')))
		# Compiling the model
		model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
		# Printing the summary of the model
		model.summary()
		return model