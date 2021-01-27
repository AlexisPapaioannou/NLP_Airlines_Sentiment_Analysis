import pandas as pd 
import re
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import TreebankWordTokenizer
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier


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

def count_vect(data, ngrams=(1, 1)):
    count_vectorizer = CountVectorizer(ngram_range=ngrams)
    emb = count_vectorizer.fit_transform(data)
    return emb, count_vectorizer

df = pd.read_csv("Tweets.csv")

#apply preprocess
df['text_cleaned']=df['text'].apply(lambda x: text_preprocess(x))

#find 10 most frequently used words
freq = pd.Series(' '.join(df['text_cleaned']).split()).value_counts()[:10]
freq = list(freq.index)
#print(freq)

# remove 10 most frequently used words
df['text_cleaned'] = df['text_cleaned'].apply(lambda w: " ".join(w for w in w.split() if w not in freq))

# keep in df only what is needed
df= df[['text_cleaned','airline_sentiment']]

#get labels
df['airline_sentiment'].replace(('neutral', 'positive', 'negative'), (0, 1, 2), inplace=True)
Y = df['airline_sentiment']
#print(Y)

#Tokenization
tokenizer = TreebankWordTokenizer() 
df["tokens"] = df['text_cleaned'].map(tokenizer.tokenize)

x = df['text_cleaned']
y = Y

df_counts, count_vectorizer = count_vect(df["text_cleaned"],ngrams=(1, 2))

X = df_counts
y = Y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)


#Logistic Regression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("\n")
print("Logistic Regression Results:")
f1score = f1_score(y_test, y_pred, average='weighted')
print(f"Counts Model f1 Score: {f1score * 100} %")
accuracy = accuracy_score(y_test, y_pred)
print(f"Counts Model Accuracy: {accuracy * 100} %")
recall_scores = recall_score(y_test,y_pred , average='weighted')
print(f"Counts Model Recall: {recall_scores * 100} %")
precision_scores = precision_score(y_test,y_pred,  average='weighted')
print(f"Counts Model Precision: {precision_scores * 100} %")
print("\n")

#SVC
model_svc = SVC(kernel='linear')
model_svc.fit(X_train, y_train)
y_pred = model_svc.predict(X_test)

print("SVC Results:")
f1score = f1_score(y_test, y_pred, average='weighted')
print(f"Counts Model f1 Score: {f1score * 100} %")
accuracy = accuracy_score(y_test, y_pred)
print(f"Counts Model Accuracy: {accuracy * 100} %")
recall_scores = recall_score(y_test,y_pred , average='weighted')
print(f"Counts Model Recall: {recall_scores * 100} %")
precision_scores = precision_score(y_test,y_pred,  average='weighted')
print(f"Counts Model Precision: {precision_scores * 100} %")
print("\n")


#Gradient Boosting Classifier
model_clf = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=3, random_state=0)
model_clf.fit(X_train, y_train)
y_pred = model_clf.predict(X_test)

print("Gradient Boosting Classifier Results:")
f1score = f1_score(y_test, y_pred, average='weighted')
print(f"Counts Model f1 Score: {f1score * 100} %")
accuracy = accuracy_score(y_test, y_pred)
print(f"Counts Model Accuracy: {accuracy * 100} %")
recall_scores = recall_score(y_test,y_pred , average='weighted')
print(f"Counts Model Recall: {recall_scores * 100} %")
precision_scores = precision_score(y_test,y_pred,  average='weighted')
print(f"Counts Model Precision: {precision_scores * 100} %")

