'''
This is the driver file which controls the complete training process
'''

from factoryModel.config import mt_config as confFile
from factoryModel.preprocessing import SentenceSplit,cleanData,TrainMaker
from factoryModel.dataLoader import textLoader
from factoryModel.models import ModelBuilding
from tensorflow.keras.callbacks import ModelCheckpoint
from factoryModel.utils.helperFunctions import *

## Define the file path to input data set
filePath = confFile.DATA_PATH

print('[INFO] Starting the preprocessing phase')

## Load the raw file and process the data
ss = SentenceSplit(50000)
cd = cleanData()
tm = TrainMaker()
# Initializing the data set loader class and then executing the processing methods
tL = textLoader(preprocessors = [ss,cd,tm])
# Load the raw data, preprocess it and create the train and test sets
eng_tokenizer,eng_vocab_size,deu_tokenizer,deu_vocab_size,text,trainX,trainY,testX,testY,eng_length,ger_length = tL.loadDoc(filePath)

print('Training shape',trainX.shape)
print('Testing shape',testX.shape)
print('Training Y shape',trainY.shape)

print('[INFO] Starting the modelling phase')

### Initiating the training phase #########
# Initialise the model
model = ModelBuilding.EncDecbuild(int(deu_vocab_size),int(eng_vocab_size),int(ger_length),int(eng_length),256)
# Define the checkpoints
checkpoint = ModelCheckpoint('model.h5',monitor = 'val_loss',verbose = 1, save_best_only = True,mode = 'min')
# Fit the model on the training data set
model.fit(trainX,trainY,epochs = 50,batch_size = 64,validation_data=(testX,testY),callbacks = [checkpoint],verbose = 2)

print('[INFO] Saving parameters and data to disk')

### Saving the tokenizers and other variables as pickle files
save_clean_data(eng_tokenizer, 'factoryModel/output/eng_tokenizer.pkl')
save_clean_data(eng_vocab_size, 'factoryModel/output/eng_vocab_size.pkl')
save_clean_data(deu_tokenizer, 'factoryModel/output/deu_tokenizer.pkl')
save_clean_data(deu_vocab_size, 'factoryModel/output/deu_vocab_size.pkl')
save_clean_data(trainX, 'factoryModel/output/trainX.pkl')
save_clean_data(trainY, 'factoryModel/output/trainY.pkl')
save_clean_data(testX, 'factoryModel/output/testX.pkl')
save_clean_data(testY, 'factoryModel/output/testY.pkl')
save_clean_data(eng_length, 'factoryModel/output/eng_length.pkl')
save_clean_data(ger_length, 'factoryModel/output/ger_length.pkl')




