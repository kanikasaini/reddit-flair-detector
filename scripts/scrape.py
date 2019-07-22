import praw
from config import Config
from pymongo import MongoClient
client = MongoClient()

db = client.reddit_precog_big

train = db.train
test = db.test


f = open('user.cfg', 'r')
config = Config(f)

reddit = praw.Reddit(username=config.username,
	   password=config.password,
	   client_id=config.client_id,
	   client_secret=config.client_secret,
	   user_agent=config.user_agent)

subreddit = reddit.subreddit('India')

flairs = ["Politics", "Non-Political", "Scheduled", "AskIndia", "Sports", "Policy", 
		  "Science & Technology", "Food", "Reddiquette"]

test_data = []
train_data = []
limit = 100
for flair in flairs:
	index = 0
	submissions = subreddit.search('flair:"' + flair + '"' , limit=limit)
	for submission in submissions:
		index = index + 1
		print(submission.title, submission.author.name)
		d = {}
		d['flair'] = flair
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
		if index > 7/10 * limit :
			test_data.append(d)
		else:
			train_data.append(d)

train.insert_many(train_data)
test.insert_many(test_data)

	

