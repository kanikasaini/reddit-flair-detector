import flask
import praw
import pickle
from sklearn.svm import SVC
import sklearn
from config import Config
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

with open('model/reddit_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
	if flask.request.method == 'GET':
		return(flask.render_template('main.html'))

	if flask.request.method == 'POST':
		flair = {1:"Politics", 2:"Non-Political", 3:"Scheduled", 4:"AskIndia", 5:"Sports", 6:"Policy", 
		  7:"Science & Technology", 8:"Food", 9:"Reddiquette"}
		posturl = flask.request.form['posturl']
		f = open('user.cfg', 'r')
		config = Config(f)
		reddit = praw.Reddit(username=config.username,
							   password=config.password,
							   client_id=config.client_id,
							   client_secret=config.client_secret,
							   user_agent=config.user_agent)


		try:
			submission = reddit.submission(url=posturl)
			d = {}
			d['title'] = submission.title
			d['author'] = submission.author.name
			d['time'] = submission.created_utc
			d['distinguished'] = submission.distinguished
			d['over_18'] = submission.over_18
			d['permalink'] = submission.permalink
			d['score'] = submission.score
			d['upvote_ratio'] = submission.upvote_ratio
			d['selftext'] = submission.selftext
			submission.comments.replace_more()
			d['comments'] = []
			for c in submission.comments.list():
				d['comments'].append(c.body)


			documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
			model1 = Doc2Vec(documents, vector_size=100, window=12, min_count=1, workers=4)
			model2 = Doc2Vec(documents, vector_size=1, window=2, min_count=1, workers=4)
			x_test = []
			x = []
			x.append(d['score'])
			x.append(d['upvote_ratio'])
			for i in model1.infer_vector(d['selftext'].split(" ")):
				x.append(i)
			for i in model1.infer_vector(d['title'].split(" ")):
				x.append(i)
			for i in model1.infer_vector(d['author'].split(" ")):
				x.append(i)
			
			x.append(d['time'])
			if(d['over_18']=='False'):
				x.append(0)
			else:
				x.append(1)

			x.append(model2.infer_vector(d['permalink']))
			comments = []
			for j in d['comments']:
				for k in j.split(" "):
					comments.append(k)

			for i in model1.infer_vector(comments):
				x.append(i)

			x_test.append(x)
			prediction = model.predict(x_test)[0]
			return flask.render_template('main.html',
			                             original_input={'post url':posturl,},
			                             result=flair[prediction],
			                             )
		except:
			return flask.render_template('main.html',
			                             original_input={'post url':posturl,},
			                             result="Invalid URL",
			                             )

#

if __name__ == '__main__':
	app.run()