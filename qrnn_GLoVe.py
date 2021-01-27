#from __future__ import print_function
import numpy as np
#np.random.seed(1337)  # for reproducibility

#from keras.preprocessing import sequence
#from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, SpatialDropout1D
#from keras.layers import LSTM, SimpleRNN, GRU
from keras.regularizers import l2
from keras.constraints import maxnorm


from qrnn import QRNN
#import numpy as np
import pandas as pd 


#import os


#import numpy as np
#import pandas as pd 


#import os


#from keras.models import Sequential
#from keras.layers import LSTM,Dense,Dropout,Embedding,BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import matplotlib.pyplot as plt
#import seaborn as sns
import re
#import nltk
from nltk.corpus import stopwords
import string

from sklearn.model_selection import train_test_split
#from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from keras.utils import to_categorical

def count_vect(data, ngrams=(1, 1)):
    count_vectorizer = CountVectorizer(ngram_range=ngrams)
    emb = count_vectorizer.fit_transform(data)
    return emb, count_vectorizer

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


df = pd.read_csv("Tweets.csv")


#apply preprocess
df['text_cleaned']=df['text'].apply(lambda x: text_preprocess(x))


#find 10 most frequently used words
freq = pd.Series(' '.join(df['text_cleaned']).split()).value_counts()[:10]
freq = list(freq.index)
print(freq)

# remove 10 most frequently used words
df['text_cleaned'] = df['text_cleaned'].apply(lambda w: " ".join(w for w in w.split() if w not in freq))

# keep in df only what is needed
df= df[['text_cleaned','airline_sentiment']]

#get labels
df['airline_sentiment'].replace(('neutral', 'positive', 'negative'), (0, 1, 2), inplace=True)
Y = df['airline_sentiment']
Y = to_categorical(df['airline_sentiment'], num_classes=3)
print(Y)

x = df['text_cleaned']
y = Y
    
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

embeddings_index = {}
f = open('C:/Users/alex/Desktop/text/project_nlp/glove.6B.50d.txt',encoding="utf8")
for line in tqdm(f):
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray([float(val) for val in values[1:]])
    embeddings_index[word] = coefs
    
f.close()

# using keras tokenizer here
token = Tokenizer(num_words=None)
max_len = 1500

token.fit_on_texts(list(X_train) + list(X_test))
xtrain_seq = token.texts_to_sequences(X_train)
xvalid_seq = token.texts_to_sequences(X_test)

#zero pad the sequences
xtrain_pad = pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = pad_sequences(xvalid_seq, maxlen=max_len)

word_index = token.word_index

embedding_matrix = np.zeros((len(word_index) + 1, 50))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


embed_dim = 128

print('Build model...')
model = Sequential()
model.add(Embedding( len(word_index) + 1, 50, weights=[embedding_matrix], input_length = embed_dim))
#model.add(SpatialDropout1D(0.2))
model.add(QRNN(128, window_size=3, dropout=0.2, 
               kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4), 
               kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10)))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(xtrain_pad, y_train,
                    epochs=10,
                    verbose=1,
                    validation_data=(xvalid_pad, y_test),
                    batch_size=32)

loss, accuracy = model.evaluate(xtrain_pad, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(xvalid_pad, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

plt.plot(history.history['accuracy'], 'ro')
plt.plot(history.history['val_accuracy'])
#plt.title('Train accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train accuracy', 'Validation accuracy'], loc='lower right')
plt.show()

plt.plot(history.history['loss'], 'ro')
plt.plot(history.history['val_loss'])
#plt.title('Train accuracy')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train loss', 'Validation loss'], loc='lower left')
plt.show()