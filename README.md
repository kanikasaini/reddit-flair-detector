Precog Task - Reddit Flair Detector

Hosted at - https://reddit-precog.herokuapp.com

In the project directory, the scripts to scrape Reddit data and train machine learning model are stored as scrape.py and train.py respectively. Analysis script to analyse the data by plotting graphs is also in the folder.
The data scraped is also stored in json format under data folder. The trained model is stored in model folder.
The detector is a flask app whose main file is app.py in the root directory, and html and image files come from templates and static folders respectively. 

Directory

	Procfile
	__pycache__
	data
		- train.json
		- test.json
	scripts
		- train.py
		- scrape.py
	templates
		- main.html
	README.md
	app.py
	model
		- reddit_model.pkl
	requirements.txt
	static
		- logo.png
	user.cfg


Dependency Libraries 

	flask
	praw
	sklearn
	config
	gensim
	gunicorn
	xgboost

Steps to run

	- activate your virtualenv
	- run 'pip3 install -r requirements.txt'
	- run 'flask run'

Machine Learning Model
	Features
		- title
		- upvotes
		- score
		- text
		- author
		- comments 
		- timestamp
		- over_18
		- hyperlink

	Used doc2vec for text to feature vector

	Accuracy on Various Classifiers
	SVM - 27.61
	MLP - 12.90
	Logistic Regression - 15.42

	The data is polluted as features are overlapping for classes, hence, the task is quite difficult to do with inbuilt scikit models.

Analysis
	- 'Non-political' has most number of upvotes.
	- 'Scheduled' has most number of comments.

References
	- https://blog.cambridgespark.com/deploying-a-machine-learning-model-to-the-web-725688b851c7
