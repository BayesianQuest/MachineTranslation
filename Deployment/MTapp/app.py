'''
This is the script for flask application
'''

from tensorflow.keras.models import load_model
from factoryModel.config import mt_config as confFile
from factoryModel.utils.helperFunctions import *
from flask import Flask,request,render_template

# Initializing the flask application
app = Flask(__name__)

## Define the file path to the model
modelPath = confFile.MODEL_PATH

# Load the model from the file path
model = load_model(modelPath)

# Get the paths for all the files and variables stored as pickle files
Eng_tokPath = confFile.ENG_TOK_PATH
Ger_tokPath = confFile.GER_TOK_PATH
Ger_length = confFile.GER_STDLEN
# Load the tokenizer from the pickle file
Eng_tokenizer = load_files(Eng_tokPath)
Ger_tokenizer = load_files(Ger_tokPath)
# Load the standard lengths
Ger_stdlen = load_files(Ger_length)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/translate', methods=['POST', 'GET'])
def get_translation():
    if request.method == 'POST':
        result = request.form
        # Get the German sentence from the Input site
        gerSentence = str(result['input_text'])
        # Converting the text into the required format for prediction
        # Step 1 : Converting to an array
        gerAr = [gerSentence]
        # Clean the input sentence
        cleanText = cleanInput(gerAr)
        # Step 2 : Converting to sequences and padding them
        # Encode the inputsentence as sequence of integers
        seq1 = encode_sequences(Ger_tokenizer, int(Ger_stdlen), cleanText)
        # Step 3 : Get the translation
        translation = generatePredictions(model,Eng_tokenizer,seq1)
        # prediction = model.predict(seq1,verbose=0)[0]

        return render_template('result.html', trans=translation)


if __name__ == '__main__':
    app.debug = True
    app.run()