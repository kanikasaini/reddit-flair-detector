from pymongo import MongoClient
import nltk
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

client = MongoClient()

db = client.reddit_precog

train = db.train
test = db.test

label = ["Politics", "Non-Political", "Scheduled", "AskIndia", "Sports", "Policy", 
		  "Science & Technology", "Food", "Reddiquette"]

flair = {"Politics": 1, "Non-Political": 2, "Scheduled": 3, "AskIndia": 4, "Sports": 5, "Policy":6, 
		  "Science & Technology": 7, "Food": 8, "Reddiquette": 9}

upvotes = [0, 0, 0, 0, 0, 0, 0, 0, 0]
comments = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for document in train.find():
	index = flair[document['flair']]
	upvotes[index-1]  = upvotes[index-1] + document['score']
	comments[index-1] = comments[index-1] + len(document['comments'])

index = np.arange(len(label))
plt.bar(index, upvotes)
plt.xlabel('Flairs', fontsize=10)
plt.ylabel('No of upvotes', fontsize=10)
plt.xticks(index, label, fontsize=8, rotation=20)
plt.title('Upvotes for each flair')
plt.show()

plt.bar(index, comments)
plt.xlabel('Flairs', fontsize=10)
plt.ylabel('No of comments', fontsize=10)
plt.xticks(index, label, fontsize=8, rotation=20)
plt.title('Comments for each flair')
plt.show()
