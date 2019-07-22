Precog Task - Reddit Flair Detector

In the project directory, the scripts to scrape Reddit data and train machine learning model are stored as scrape.py and train.py respectively. 
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
