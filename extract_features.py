
import os
import sys
import nltk
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
from xml.etree.cElementTree import iterparse

#Tokenization
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return tokens

#Stemming
def stem(tokens):
    stemmed = ""
    for item in tokens:
        stemmed = stemmed + (SnowballStemmer('english').stem(item)) + " "
    return stemmed

#Building vocabulary
tokenDictionary = {}
def buildVocabulary(documents):
    index = 0
    for text in documents:
        index = index + 1
        text = text.lower()
        try:
            text = text.translate(None, string.punctuation)
        except Exception, e:
            pass

        tokens = tokenize(text)
        stemmed = stem(tokens)

        tokenDictionary[index] = stemmed

#Extract text from file
file = open('./en/books/train.review', 'r')

#List of reviews
documents = list()

#Number of reviews
itemCount = 0
try:
    for event, element in iterparse(file):
        if element.tag == "category":
            category = element.text
        elif element.tag == "rating":
            rating = float(element.text)
        elif element.tag == "date":
            date = element.text
        elif element.tag == "text":
            text = element.text
            documents.append(text)
        elif element.tag == "summary":
            summary = element.text
        itemCount = itemCount + 1
except SyntaxError, syntaxError:
    print syntaxError

#Build vocabulary
buildVocabulary(documents)

tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(tokenDictionary.values())

for text in documents:
    response = tfidf.transform([text])
    print response, "\n"



