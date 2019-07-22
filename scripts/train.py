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

client = MongoClient()

db = client.reddit_precog

train = db.train
test = db.test

x_train = []
y_train = []
x_test = []
y_test = []

flair = {"Politics": 1, "Non-Political": 2, "Scheduled": 3, "AskIndia": 4, "Sports": 5, "Policy":6, 
		  "Science & Technology": 7, "Food": 8, "Reddiquette": 9}

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
model1 = Doc2Vec(documents, vector_size=100, window=12, min_count=1, workers=4)
model2 = Doc2Vec(documents, vector_size=1, window=2, min_count=1, workers=4)

for document in train.find():
	x = []
	y = flair[document['flair']]
	x.append(document['score'])
	x.append(document['upvote_ratio'])
	for i in model1.infer_vector(document['selftext'].split(" ")):
		x.append(i)
	for i in model1.infer_vector(document['title'].split(" ")):
		x.append(i)
	for i in model1.infer_vector(document['author'].split(" ")):
		x.append(i)
	
	x.append(document['time'])
	if(document['over_18']=='False'):
		x.append(0)
	else:
		x.append(1)

	x.append(model2.infer_vector(document['permalink']))
	comments = []
	for j in document['comments']:
		for k in j.split(" "):
			comments.append(k)

	for i in model1.infer_vector(comments):
		x.append(i)

	x_train.append(x)
	y_train.append(y)

for document in test.find():
	x = []
	y = flair[document['flair']]
	x.append(document['score'])
	x.append(document['upvote_ratio'])
	for i in model1.infer_vector(document['selftext'].split(" ")):
		x.append(i)
	for i in model1.infer_vector(document['title'].split(" ")):
		x.append(i)
	for i in model1.infer_vector(document['author'].split(" ")):
		x.append(i)
	
	x.append(document['time'])
	if(document['over_18']=='False'):
		x.append(0)
	else:
		x.append(1)

	x.append(model2.infer_vector(document['permalink']))
	comments = []
	for j in document['comments']:
		for k in j.split(" "):
			comments.append(k)

	for i in model1.infer_vector(comments):
		x.append(i)

	x_test.append(x)
	y_test.append(y)

#print(x_train)
#print(y_train)
clfsvm = SVC(gamma=0.000000001, C=1)
clfsvm.fit(x_train, y_train) 

#clf = 
y_pred = clfsvm.predict(x_test)
print(classification_report(y_test, y_pred))

print("Test Accuracy", accuracy_score(y_test, y_pred))
y_pred = clfsvm.predict(x_train)
print("Train Accuracy", accuracy_score(y_train, y_pred))
print(y_pred)


clf = MLPClassifier(hidden_layer_sizes=(100, 20), solver='lbfgs', alpha=1e-9)
clf.fit(x_train, y_train) 

#clf = 
y_pred = clf.predict(x_test)
print("Test Accuracy", accuracy_score(y_test, y_pred))
y_pred = clf.predict(x_train)
print("Train Accuracy", accuracy_score(y_train, y_pred))

clf = LogisticRegression()
clf.fit(x_train, y_train) 

#clf = 
y_pred = clf.predict(x_test)
print("Test Accuracy", accuracy_score(y_test, y_pred))
y_pred = clf.predict(x_train)
print("Train Accuracy", accuracy_score(y_train, y_pred))


outfile = open('../model/reddit_model.pkl','wb')
pickle.dump(clfsvm, outfile)
outfile.close()

