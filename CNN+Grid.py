# Airlines sentiment
# Konstantinos Serderidis AEM 46
# ======================================================================================================================
# Credits:https://colab.research.google.com/drive/1q2RvpX5No5NQ5P-J4INtF6Z7hAx7R_Fw

# installations, if not already there
#!pip install tensorflow


# imports
import os
import re
import shutil
import string

# Time
import time
import datetime

# Numerical
import numpy as np
import pandas as pd

# Tools
import itertools
from collections import Counter

# NLP
import re
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost

# Preprocessing
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.utils import class_weight as cw
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# Model Selection
from sklearn.model_selection import train_test_split
from sklearn import datasets,model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, plot_importance


# Evaluation Metrics
from sklearn import metrics 
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix,classification_report

# Tensorflow-Keras
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional,Flatten, GlobalAveragePooling1D, Dropout, MaxPooling1D, Input, Conv2D, MaxPool2D
from tensorflow.keras.layers import Input, Add, Concatenate, Dense, Activation, BatchNormalization, Dropout, SpatialDropout1D, Flatten, Reshape

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('Tweets.csv')

#preprocess function
def text_preprocess(tweet):
    letters = re.sub("^a-zA-Z"," ",tweet)
    http = re.sub(r'http\S+', '',letters)
    references = re.sub(r'@\w+', '', http)
    signs=re.sub(r'#\w+', '', references)
    p = re.sub(r'[^\w\s]','',signs)
    words = p.lower().split()
    stops = set(stopwords.words("english"))
    whitelist = ["n't", "not", "no"," "]
    punct = string.punctuation+"“”’" # add unusual apostrophes
    meaningful_words = [w for w in words if (w not in stops or w not in punct or w in whitelist)]
    return( " ".join(meaningful_words))

#apply preprocess
df['text_cleaned']=df['text'].apply(lambda x: text_preprocess(x))

# remove 10 most frequently used words
freq = pd.Series(' '.join(df['text_cleaned']).split()).value_counts()[:10]
freq = list(freq.index)
df['text_cleaned'] = df['text_cleaned'].apply(lambda w: " ".join(w for w in w.split() if w not in freq))

# keep in df only what is needed
df= df[['text_cleaned','airline_sentiment']]

#get labels
df['airline_sentiment'].replace(('neutral', 'positive', 'negative'), (0, 1, 2), inplace=True)
Y = df['airline_sentiment']




from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# functions 

def build_model(num_filters, kernel_size, vocab_size, embedding_dim, max_length):
    
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=max_length))
    model.add(Dropout(0.4))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu', padding='same'))
    model.add(layers.GlobalMaxPooling1D())
    #model.add(layers.MaxPooling1D())
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation='softmax'))
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # change the learning_rate parameter to assess the loss
    model.compile(optimizer=adam_optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model



def metric_evaluation(history, metric_id, epochs):
    '''
    Function to evaluate a trained model on a selected metric. 
   
    Input parameters:
        history : model training history
        metric_id: loss or accuracy
    Output:
        line chart with epochs of x-axis and metric on y-axis
    '''
    metric = history.history[metric_id]
    validation_metric = history.history['val_' + metric_id]

    epochs_r = range(1, epochs + 1)

    plt.plot(epochs_r, metric, 'bo', label='Train ' + metric_id)
    plt.plot(epochs_r, validation_metric, 'b', label='Validation ' + metric_id)
    plt.legend()
    plt.show()


# start 
X = df["text_cleaned"]
y = Y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=37)


tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

epochs= 10
max_length = 100
embedding_dim = 50

padded_train = pad_sequences(sequences_train, padding='post', truncating='post', maxlen=max_length)
padded_test = pad_sequences(sequences_test, padding='post', truncating='post', maxlen=max_length)

#build a grid search
param_grid = dict(num_filters=[128, 64, 32],
                  kernel_size=[3, 5, 7],
                  vocab_size=[vocab_size], 
                  embedding_dim=[embedding_dim],
                  max_length=[max_length])

model = KerasClassifier(build_fn=build_model,epochs=epochs, verbose=1)
#grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=2, verbose=1, n_iter=6)
grid = GridSearchCV(estimator=model, param_grid = param_grid, cv=2)

#grid_result = grid.fit(padded_train, y_train)
grid_result = grid.fit(padded_train, y_train, epochs=epochs, validation_split=0.2, verbose=1)

# Evaluate testing set
test_accuracy = grid.score(padded_test, y_test)
s = ('Accuracy : ' '{:.5f}\n{}\nTest Accuracy : {:.5f}\n\n')
output_string = s.format(grid_result.best_score_, grid_result.best_params_, test_accuracy)
print(output_string)


# results for the best grid parameters

num_filters = grid_result.best_params_['num_filters']
kernel_size = grid_result.best_params_['kernel_size']          

#build and fit model for the best grid result

model = build_model(num_filters, kernel_size, vocab_size, embedding_dim, max_length)

#history = model.fit(padded_train, y_train, epochs=epochs, validation_split=0.2, verbose=1) 
history = model.fit(padded_train, y_train,epochs=epochs,verbose=1,validation_data=(padded_test, y_test))
   
# Plots metrics variation over epochs 
metric_evaluation(history, 'accuracy', epochs)
metric_evaluation(history, 'loss', epochs)
