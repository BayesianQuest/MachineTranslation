'''
This is the configuration file for storing all the application parameters
'''

import os
from os import path


# This is the base path to the Machine Translation folder
BASE_PATH = '/media/acer/7DC832E057A5BDB1/JMJTL/Tomslabs/BayesianQuest/MT/MTapp'
# Define the path where the model is saved
MODEL_PATH = path.sep.join([BASE_PATH,'factoryModel/output/model.h5'])
# Defin the path to the tokenizer
ENG_TOK_PATH = path.sep.join([BASE_PATH,'factoryModel/output/eng_tokenizer.pkl'])
GER_TOK_PATH = path.sep.join([BASE_PATH,'factoryModel/output/deu_tokenizer.pkl'])
# Path to Standard lengths of German and English sentences
GER_STDLEN = path.sep.join([BASE_PATH,'factoryModel/output/ger_length.pkl'])
ENG_STDLEN = path.sep.join([BASE_PATH,'factoryModel/output/eng_length.pkl'])

######## German Sentence for Translation ###############

GER_SENTENCE = 'heute ist ein guter Tag'

