from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Embedding, LSTM, Dense, Dropout
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import sys

sys.path.append('E:/Project/AnalysisOnMovie/Tokeniz')  
from Token import PreparData, Tokenize, DataSequences, paddingData

path = './ProcessedData/processed_imdb_reviews.csv'

# Load data
Data = pd.read_csv(path)
x = PreparData(path)
y = Data['sentiment']
y = y.map({'positive': 1, 'negative': 0})

# Tokenize and pad sequences
Token = Tokenize(x)
SeqData = DataSequences(Token, x)
padding = paddingData(SeqData)

# Train/test split
XTrain, XTest, YTrain, YTest = train_test_split(padding, y, test_size=0.2)

# Model parameters
vocab_size = len(Token.word_index) + 1
embedding_dim = 100
max_sequence_length = 200

# Build the model
model = Sequential()
model.add(InputLayer(input_shape=(max_sequence_length,)))
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))

model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=128))

model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
with tf.device('/GPU:0'):
    model.fit(XTrain, YTrain, batch_size=128, epochs=10, validation_split=0.2)

# Evaluate the model
results = model.evaluate(XTest, YTest, batch_size=128)
print("Test loss, Test accuracy:", results)

# Predictions
predictions = model.predict(XTest)
pred_labels = (predictions > 0.5).astype("int32")

# Save the model
model.save('./My_model/LSTM_Model.h5')
print("Model saved!")

# Classifiaction_report 
report = classification_report(pred_labels, YTest)
print(report)

# Predictions and confusion matrix
ConfusionMatrixDisplay.from_predictions(YTest, pred_labels)
plt.show()
