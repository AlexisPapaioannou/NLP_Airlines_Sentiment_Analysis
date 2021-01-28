# NLP Airlines sentiment analysis
# Kontantinos Serderidis AEM 46
# ======================================================================================================================
# installations, if not already there
#!pip install tensorflow

# basic imports
import os
import re
import shutil
import string
import seaborn as sns
import matplotlib.pyplot as plt

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

# Preprocessing
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.utils import class_weight as cw
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# Model Selection
import xgboost
from xgboost import XGBClassifier, plot_importance
from sklearn import datasets,model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


# Evaluation Metrics
from sklearn import metrics 
from sklearn.metrics import f1_score,recall_score, accuracy_score,precision_score, confusion_matrix,classification_report, log_loss

# Tensorflow-Keras
import tensorflow as tf

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional,Flatten, GlobalMaxPooling1D,  GlobalAveragePooling1D, Dropout, MaxPooling1D, Input,  Conv1D, Conv2D, MaxPool2D
from tensorflow.keras.layers import Input, Add, Concatenate, Dense, Activation, BatchNormalization, Dropout, SpatialDropout1D, Flatten, Reshape
from tensorflow.keras.optimizers import RMSprop, Adam


# read input file
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


# functions

def basicCNN(num_filters, kernel_size, vocab_size, embedding_dim, max_length):
    '''
    Build a basic CNN model 
   
    Input parameters:num_filters, kernel_size, vocab_size, embedding_dim,max_length, embedding_matrix
    Output:model
   '''
   
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],input_length=max_length))
    model.add(Dropout(0.4))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu',padding='same'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation='softmax'))

    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) #  change the learning_rate parameter to assess the loss
    model.compile(optimizer=adam_optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    
    return model

def basicCNNGlove(num_filters, kernel_size, vocab_size, embedding_dim, max_length, embedding_matrix, trainable=False):
    '''
    Build a basic CNN model with Glove
   
    Input parameters:num_filters, kernel_size, vocab_size, embedding_dim,max_length, embedding_matrix
    Output:model
   '''
   
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],input_length=max_length))
    model.add(Dropout(0.4))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu',padding='same'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation='softmax'))

    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) #  change the learning_rate parameter to assess the loss
    model.compile(optimizer=adam_optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    
    return model


def textCNN1D(num_filters, kernel_size, vocab_size, embedding_dim, max_length, embedding_matrix, trainable=False):
    '''
    Build a textCNN model vaiant based on Conv1D
    
    reference: Yoon Kim. 2014.  Convolutional neural networks for sentence classification.arXivpreprint arXiv:1408.5882(2014)
    
    Input parameters:num_filters, kernel_size, vocab_size, embedding_dim,max_length, embedding_matrix
    Output:model
   '''
    filter_sizes = [3,4,5]
    
    inp = Input(shape=(max_length,))
    x = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix])(inp)
    convs = []

    for filter_size in filter_sizes:
        conv = Conv1D(num_filters, kernel_size=filter_size, activation='relu')(x)
        bn = BatchNormalization()(conv)
        drop = Dropout(0.2)(bn)
        pool = GlobalMaxPooling1D()(drop)
        flat = Flatten()(pool)
        convs.append(pool)
      
    conc = Concatenate(axis=1)(convs)   
    #flat = Flatten()(conc)
    
    drop = Dropout(0.5)(conc)
    
    dense= Dense(128, activation="relu")(drop)
    bn = BatchNormalization()(dense)
    drop2 = Dropout(0.5)(bn)

    outp = Dense(3, activation="softmax")(drop2)

    model = Model(inputs=inp, outputs=outp)
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) #  change the learning_rate parameter to assess the loss
    model.compile(optimizer=adam_optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    #model.summary()
    
    return model


def textCNN2Da(num_filters, kernel_size, vocab_size, embedding_dim, max_length, embedding_matrix, trainable=False):
    
    '''
    Build a textCNN model variant based on Conv2D and concatenated layers v1
    
    reference: Linyuan Gong and Ruyi Ji. 2018. What Does a TextCNN Learn? arXiv preprintarXiv:1801.06287(2018).
   
    Input parameters:num_filters, kernel_size, vocab_size, embedding_dim,max_length, embedding_matrix
    Output:model
   '''
    
    filter_sizes = [3,4,5]
    
    inp = Input(shape=(max_length,))
    x = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix])(inp)
    
    reshape = Reshape((max_length, embedding_dim, 1))(x)
    
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim))(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim))(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim))(reshape)

    
    bn_0 =BatchNormalization(axis=3)(conv_0)
    bn_1 =BatchNormalization(axis=3)(conv_1)
    bn_2 =BatchNormalization(axis=3)(conv_2)
    
    act_0 =Activation("relu")(bn_0)
    act_1 =Activation("relu")(bn_1)
    act_2 =Activation("relu")(bn_2)
        
    maxpool_0 = MaxPool2D(pool_size=(max_length - filter_sizes[0] + 1, 1))(act_0)
    maxpool_1 = MaxPool2D(pool_size=(max_length - filter_sizes[1] + 1, 1))(act_1)
    maxpool_2 = MaxPool2D(pool_size=(max_length - filter_sizes[2] + 1, 1))(act_2)
        
    
    conc = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2] )   
    flat = Flatten()(conc)
    drop = Dropout(0.5)(flat)
    
    dense= Dense(128, activation="relu")(drop)
    drop = Dropout(0.2)(dense)

    outp = Dense(3, activation="softmax")(drop)

    model = Model(inputs=inp, outputs=outp)
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) #  change the learning_rate parameter to assess the loss
    model.compile(optimizer=adam_optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   
    model.summary()
    
    return model


def textCNN2Db(num_filters, kernel_size, vocab_size, embedding_dim, max_length, embedding_matrix, trainable=False):
    
    '''
    Build a textCNN model variant based on Conv2D and concatenated layers v2
    
    reference: Linyuan Gong and Ruyi Ji. 2018. What Does a TextCNN Learn? arXiv preprintarXiv:1801.06287(2018).
   
    Input parameters:num_filters, kernel_size, vocab_size, embedding_dim,max_length, embedding_matrix
    Output:model
    '''
    
    filter_sizes = [3,4,5]
    
    inp = Input(shape=(max_length,))
    x = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)(inp)
    #x = BatchNormalization(x)
    reshape = Reshape((max_length, embedding_dim, 1))(x)

    max_pool = []
    bn = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embedding_dim), kernel_initializer='he_normal', activation='relu')(reshape)
        max_pool.append(MaxPool2D(pool_size=(max_length - filter_sizes[i] + 1, 1))(conv))

    conc = Concatenate(axis=1)(max_pool)   
    flat = Flatten()(conc)
    drop = Dropout(0.5)(flat)

    #outp = Dense(128, activation="relu")(drop)
    #drop = Dropout(0.5)(outp)

    outp = Dense(3, activation="softmax")(drop)
    
    model = Model(inputs=inp, outputs=outp)
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) #  change the learning_rate parameter to assess the loss
    model.compile(optimizer=adam_optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model


def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath, encoding="utf8") as f:
        
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


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
    plt.xlabel('epochs')
    plt.ylabel('metrics')
    plt.legend()
    plt.show()


#==========================================TF-IDF=====================================================================================

#tfidf_vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000, min_df=5, max_df=0.7,ngram_range=(1,2)) 
#tfidf = TfidfVectorizer(max_features=10000)
tfidf = TfidfVectorizer()

X = tfidf.fit_transform(df["text_cleaned"])
y = Y

X_train_tfid, X_test, y_train_tfid, y_test = train_test_split(X, y, test_size=0.2,random_state=37)

from sklearn.linear_model import LogisticRegression

model_lg = LogisticRegression(max_iter = 10000)
model_lg.fit(X_train_tfid, y_train_tfid)

y_pred = model_lg.predict(X_test)

f1score = f1_score(y_test, y_pred, average='weighted')
print(f"Tf-idf LR Model F1 Score: {f1score * 100} %")
accuracy = accuracy_score(y_test, y_pred)
print(f"Tf-idf LR Model Accuracy: {accuracy * 100} %")
recall = metrics.recall_score(y_test, y_pred, average= 'macro')
print(f"Tf-idf LR Model Recall Score: {recall * 100} %")
precision = metrics.precision_score(y_test, y_pred, average= 'macro')
print(f"Tf-idf LR Model Precision Score: {precision * 100} %")


pred = model_lg.predict_proba(X_test)
loss = log_loss(y_test, pred) 
print(f"Tf-idf LR Model loss: {loss * 100} %")

#==========================================SVM=====================================================================================
svc = SVC(gamma = 'auto')
model_svc = SVC(kernel='linear', probability=True)
model_svc.fit(X_train_tfid, y_train_tfid)

y_pred = model_svc.predict(X_test)

f1score = f1_score(y_test, y_pred, average='weighted')
print(f"Tf-idf SVM Model F1 Score: {f1score * 100} %")
accuracy = accuracy_score(y_test, y_pred)
print(f"Tf-idf SVM Model Accuracy: {accuracy * 100} %")
recall = metrics.recall_score(y_test, y_pred, average= 'macro')
print(f"Tf-idf SVM Model Recall Score: {recall * 100} %")
precision = metrics.precision_score(y_test, y_pred, average= 'macro')
print(f"Tf-idf SVM Model Precision Score: {precision * 100} %")


pred = model_svc.predict_proba(X_test)
loss = log_loss(y_test,pred) 
print(f"Tf-idf SVM Model loss: {loss * 100} %")


#==========================================GB=====================================================================================#

model_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
model_clf.fit(X_train_tfid, y_train_tfid)

y_pred = model_clf.predict(X_test)


f1score = f1_score(y_test, y_pred, average='weighted')
print(f"Tf-idf GB Model F1 Score: {f1score * 100} %")
accuracy = accuracy_score(y_test, y_pred)
print(f"Tf-idf GB Model Accuracy: {accuracy * 100} %")
recall = metrics.recall_score(y_test, y_pred, average= 'macro')
print(f"Tf-idf GB Model Recall Score: {recall * 100} %")
precision = metrics.precision_score(y_test, y_pred, average= 'macro')
print(f"Tf-idf GB Model Precision Score: {precision * 100} %")

pred = model_clf.predict_proba(X_test)
loss = log_loss(y_test,pred) 
print(f"Tf-idf GB Model loss: {loss * 100} %")



#==========================================CNN=====================================================================================#

X = df["text_cleaned"]
y = Y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=37)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)

#tokenizer = Tokenizer(num_words=10000)
#tokenizer.fit_on_texts(X)

#print(y_test())
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

#sequences = tokenizer.texts_to_sequences(X)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

# input parameters
max_length = 64
epochs = 10
embedding_dim = 50
num_filters= 64
kernel_size= 5

embedding_matrix = create_embedding_matrix('glove.twitter.27B.50d.txt',tokenizer.word_index, embedding_dim)

padded_train = pad_sequences(sequences_train, padding='post', truncating='post', maxlen=max_length)
padded_test = pad_sequences(sequences_test, padding='post', truncating='post', maxlen=max_length)

#X = pad_sequences(sequences, padding='post', truncating='post', maxlen=max_length)
# to pad complete X instead

#select, build and fit models

#model = basicCNN(num_filters, kernel_size, vocab_size, embedding_dim, max_length)
model = basicCNNGlove(num_filters, kernel_size, vocab_size, embedding_dim, max_length,embedding_matrix)
#model = textCNN1D(num_filters, kernel_size, vocab_size, embedding_dim, max_length,embedding_matrix)
#model = textCNN2Da(num_filters, kernel_size, vocab_size, embedding_dim, max_length,embedding_matrix)
#model = textCNN2Db(num_filters, kernel_size, vocab_size, embedding_dim, max_length,embedding_matrix)


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=37)
# to split later instead

#history = model.fit(padded_train, y_train, epochs=epochs, validation_split=0.2, verbose=1) 
history = model.fit(padded_train, y_train,epochs=epochs,verbose=1,validation_data=(padded_test, y_test))


# Plots metrics variation over epochs 
metric_evaluation(history,'accuracy', epochs)
metric_evaluation(history,'loss', epochs)

    
loss, accuracy = model.evaluate(padded_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(padded_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

y_pred = model.predict(padded_test)
y_pred = [np.argmax(k, axis=None, out=None) for k in y_pred]

# Print metrics
print("The f1 score is:", f1_score(y_test, y_pred,average='weighted'))
print("The recall score is:", recall_score(y_test, y_pred,average='weighted'))
print("The precision score is:", precision_score(y_test, y_pred,average='weighted'))

print(metrics.classification_report(y_test, y_pred, digits=3))
print(metrics.confusion_matrix(y_test, y_pred))

