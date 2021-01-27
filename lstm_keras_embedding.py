import numpy as np
import pandas as pd 

from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,Embedding,BatchNormalization,Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import matplotlib.pyplot as plt
import re

from nltk.corpus import stopwords
import string
from sklearn.metrics import f1_score, recall_score,precision_score

from sklearn.model_selection import train_test_split
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
    
#sentences_train, sentences_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1000)

#Tokenization


max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(x.values)
X = tokenizer.texts_to_sequences(x.values)
X = pad_sequences(X)


embedding_dim = 128
input_len = X.shape[1]
input_len = 26
max_features = 2000

model = Sequential()
model.add(Embedding(max_features, embedding_dim, input_length = input_len))
model.add(Dropout(0.5))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

history = model.fit(X_train, y_train,
                    epochs=10,
                    verbose=1,
                    validation_data=(X_test, y_test),
                    batch_size=128)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


y_pred = model.predict(X_test)

y_pred = [np.argmax(k, axis=None, out=None) for k in y_pred]
y_test = [np.argmax(l, axis=None, out=None) for l in y_test]

print("The f1 score is:", f1_score(y_test, y_pred,average='weighted'))
print("The recall score is:", recall_score(y_test, y_pred,average='weighted'))
print("The precision score is:", precision_score(y_test, y_pred,average='weighted'))



print(history.history['accuracy'])
print(history.history['val_accuracy'])

plt.plot(history.history['accuracy'], 'ro')
plt.plot(history.history['val_accuracy'])
#plt.title('Train accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train accuracy', 'Validation accuracy'], loc='lower right')
plt.show()

print(history.history['loss'])
print(history.history['val_loss'])

plt.plot(history.history['loss'], 'ro')
plt.plot(history.history['val_loss'])
#plt.title('Train accuracy')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train loss', 'Validation loss'], loc='lower left')
plt.show()