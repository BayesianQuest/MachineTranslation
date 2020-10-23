'''
This is the driver file for the inference process
'''

from tensorflow.keras.models import load_model
from factoryModel.config import mt_config as confFile
from factoryModel.utils.helperFunctions import *

## Define the file path to the model
modelPath = confFile.MODEL_PATH

# Load the model from the file path
#model = load_model(modelPath)
model = load_model('/media/acer/7DC832E057A5BDB1/JMJTL/Tomslabs/BayesianQuest/MT/MTapp/factoryModel/output/model.h5')

# Get the paths for all the files and variables stored as pickle files
Eng_tokPath = confFile.ENG_TOK_PATH
Ger_tokPath = confFile.GER_TOK_PATH
testxPath = confFile.TEST_X
testyPath = confFile.TEST_Y
Ger_length = confFile.GER_STDLEN
# Load the tokenizer from the pickle file
Eng_tokenizer = load_files(Eng_tokPath)
Ger_tokenizer = load_files(Ger_tokPath)
# Load the standard lengths
Ger_stdlen = load_files(Ger_length)
# Load the test sets
testX = load_files(testxPath)
testY = load_files(testyPath)

# Generate predictions
predSent = generatePredictions(model,Eng_tokenizer,testX[20:30,:])

for i in range(len(testY[20:30])):
    targetY = Convertsequence(Eng_tokenizer,testY[i:i+1][0])
    print("Original sentence : {} :: Prediction : {}".format([targetY],[predSent[i]]))

############# Prediction of your Own sentences ##################

# Get the input sentence from the config file
inputSentence = [confFile.GER_SENTENCE]

# Clean the input sentence
cleanText = cleanInput(inputSentence)

# Encode the inputsentence as sequence of integers
seq1 = encode_sequences(Ger_tokenizer,int(Ger_stdlen),cleanText)

print("[INFO] .... Predicting on own sentences...")

# Generate the prediction
predSent = generatePredictions(model,Eng_tokenizer,seq1)
print("Original sentence : {} :: Prediction : {}".format([cleanText[0]],predSent))















