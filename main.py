import sys
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Word2vec
import gensim

# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle
import itertools

from preprocess import build_coin, WIN_SIZE, OOV_DATASET, NO_OOV_DATASET

# Set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# DATASET
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
OOV_AUX_COLUMNS = ['win' + str(w) for w in WIN_SIZE]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.8

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# WORD2VEC 
W2V_SIZE = 400
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 1024

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# EXPORT
KERAS_MODEL = "model.h5"
WORD2VEC_MODEL = "../wiki_all.model/wiki_all.sent.split.model"
TOKENIZER_MODEL = "tokenizer.pkl"
ENCODER_MODEL = "encoder.pkl"

COIN_PATH = 'COIN.pickle'
if not os.path.exists(COIN_PATH):
    build_coin()
with open(COIN_PATH, 'rb') as f:
    coinVec = pickle.load(f)

dataset_filename = os.listdir("input")[0]
dataset_path = os.path.join("input",dataset_filename)

decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
def decode_sentiment(label):
    return decode_map[int(label)]
DEBUG = False
df_train = pd.read_csv(NO_OOV_DATASET, encoding =DATASET_ENCODING , names=DATASET_COLUMNS)
df_train = df_train.sample(frac=1).reset_index(drop=True)
df_test = pd.read_csv(OOV_DATASET, encoding =DATASET_ENCODING , names=DATASET_COLUMNS+OOV_AUX_COLUMNS)
if DEBUG:
    df_train = df_train[:1000]
    df_test = df_test.sample(frac=1).reset_index(drop=True)[:1000]
w2v_model = gensim.models.word2vec.Word2Vec.load(WORD2VEC_MODEL)
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(df_train.text)
for col in OOV_AUX_COLUMNS:
    tokenizer.fit_on_texts(df_test[col])
    
vocab_size = len(tokenizer.word_index) + 1

x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=SEQUENCE_LENGTH)
x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=SEQUENCE_LENGTH)
x_test_coin = [pad_sequences(tokenizer.texts_to_sequences(df_test[col]), maxlen=SEQUENCE_LENGTH) for col in OOV_AUX_COLUMNS]
labels = df_train.target.unique().tolist()
labels.append(NEUTRAL)

encoder = LabelEncoder()
encoder.fit(df_train.target.tolist())

y_train = encoder.transform(df_train.target.tolist())
y_test = encoder.transform(df_test.target.tolist())

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# print("y_train",y_train.shape)
# print("y_test",y_test.shape)

# print("x_train", x_train.shape)
# print("y_train", y_train.shape)
# print()
# print("x_test", x_test.shape)
# print("y_test", y_test.shape)

embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
    else:
        # word is an oov term, use its coin vector
        if '_' not in word:
            continue
        word, win = word.split('_')
        if word not in coinVec:
            logging.warning(f'Missing oov term {word} in COIN vectors.')
            continue
        win = int(win)
        if win in coinVec[word]:
            embedding_matrix[i] = coinVec[word][win]

embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH, trainable=False)

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.3))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]

history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=callbacks)
with open('score.log', 'a') as f:
    score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    f.write("Baseline accuracy: {:.4f}\n".format(score[1]))
    logging.info("Baseline accuracy: {:.4f}\n".format(score[1]))
    for col, x_test in zip(OOV_AUX_COLUMNS, x_test_coin):
        score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
        f.write("{} accuracy: {:.4f}\n".format(col, score[1]))
        logging.info("{} accuracy: {:.4f}\n".format(col, score[1]))


# y_pred_1d = []
# y_test_1d = list(df_test.target)
# scores = model.predict(x_test, verbose=1, batch_size=8000)
# y_pred_1d = [decode_sentiment(score, include_neutral=False) for score in scores]
# with open('score.log', 'a') as f:
#     f.write("\t ACCURACY: {}\n".format(score[1]))
#     f.write("\t Classification report: " + str(classification_report(y_test_1d, y_pred_1d)) + "\n")
#     f.write("\t Accuracy score: " + str(accuracy_score(y_test_1d, y_pred_1d)) + "\n")

