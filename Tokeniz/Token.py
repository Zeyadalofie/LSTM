import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd


def PreparData(text):
    # Step 1: Read the CSV file
    Data = pd.read_csv(text)
    DropData = Data["Processed text"].values  # Extract the processed text column
    return DropData
def Tokenize(Data):
    # Step 2: Tokenize the text
    tokenizer = Tokenizer(num_words = 5000, oov_token='<OOV>') # Consider limiting to top 5000 words
    tokenizer.fit_on_texts(Data)
    return tokenizer  # Return the tokenizer object
def DataSequences(tokenizer, reviews):
    # Step 3: Convert reviews to tokenized sequences using the same tokenizer
    sequences = tokenizer.texts_to_sequences(reviews)  # Convert reviews to sequences
    return sequences  # Return the tokenized sequences

def paddingData(sequences):
     
    max_len = 200  # Set a reasonable maxlen manually, instead of using max sequence length
    
    # Pad sequences
    padded_sequence = pad_sequences(sequences, maxlen=max_len, padding='post')
    
    return padded_sequence
