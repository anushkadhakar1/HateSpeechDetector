import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import re
import string
import nltk
from nltk.util import pr
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Initialize the stemmer
stemmer = SnowballStemmer("english")

# Load English stopwords
stopwords_set = set(stopwords.words("english"))

data = pd.read_csv('twitter_data.csv')
df = pd.DataFrame(data)

df['labels'] = df['class'].map({0: "Hate speech detected" , 1: "Offensive language detected", 2: "No hate and offensive speech"})

df = df[['tweet','labels']]

df.dropna(subset=["tweet", "labels"], inplace=True)  # Drop rows with NaN in 'tweet' or 'labels'
df["tweet"] = df["tweet"].fillna("")  # Optional: Fill NaN text with an empty string
df["labels"] = df["labels"].fillna(0)  # Optional: Replace missing labels with a default value

def clean(text):
    
    text =str(text).lower()
    text = re.sub('\[.*?\]','' ,text)
    text = re.sub('https?://\S+|www\.\S+','' ,text)
    text = re.sub('<.*?>+','' ,text)
    # text = re.sub('[%]',%re.escape(string.punctuation),'' ,text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub('\n','' ,text)
    text = re.sub('\w*\d\w*','' ,text)
    text = [word for word in text.split()if word not in stopwords_set]
    text = " ".join(text)
    return text
df["tweet"] = df["tweet"].apply(clean)

x = np.array(df["tweet"])
y = np.array(df["labels"])
cv = CountVectorizer()
x = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state = 42)
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
