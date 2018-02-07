#! python2
# -*- coding: utf-8 -*-
import pprint
import pymongo
import datetime
from pymongo import MongoClient

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation, NMF
import os
import re

from nltk import word_tokenize, pos_tag

import progressbar

import spacy
nlp = spacy.load('en_core_web_sm') 

db_name = 'twitter'
col_name = 'DataEN'

col_tag_name = 'topic'
col_tag_name_2 = 'ex'

no_features = 5000
no_topics = 100
no_top_words = 10

ignore_under = 20

class MyPrettyPrinter(pprint.PrettyPrinter):
    def format(self, object, context, maxlevels, level):
        if isinstance(object, unicode):
            return (object.encode('thai'), True, False)
        return pprint.PrettyPrinter.format(self, object, context, maxlevels, level)

def go_print( input ):
	MyPrettyPrinter().pprint(input)
	# ppp = pprint.PrettyPrinter(indent=4)
	# ppp.pprint(input)
	return;

def get_midnight(time):
	return time.replace(minute=0, hour=0, second=0, microsecond=0)

def get_thai_midnight(time):
	out = time + datetime.timedelta(hours=7)
	out = out.replace(minute=0, hour=0, second=0, microsecond=0) - datetime.timedelta(hours=7)
	return out

def get_NMF(documents):
	tfidf_vectorizer = TfidfVectorizer(max_df=0.80, min_df=2)
	tfidf = tfidf_vectorizer.fit_transform(documents)
	nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)

	tfidf_feature_names = tfidf_vectorizer.get_feature_names()
	topics_set = []
	for topic_idx, topic in enumerate(nmf.components_):
		topics_set.append(",".join([ tfidf_feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]) )
	score = nmf.transform(tfidf)
	topic = score.argmax(axis=1)
	return topics_set, topic

def get_LDA(documents):
	tf_vectorizer = CountVectorizer(max_df=0.80, min_df=2)
	tf = tf_vectorizer.fit_transform(documents)
	lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
	
	tf_feature_names = tf_vectorizer.get_feature_names()
	topics_set = []
	for topic_idx, topic in enumerate(lda.components_):
		topics_set.append(",".join([ tf_feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]) )
	score = lda.transform(tf)
	topic = score.argmax(axis=1)
	return topics_set, topic

def preprocess(documents):
	out = []

	i = 0
	for text in documents:
		# remove mention & url
		re_text = re.sub(r'(@[^\s]+)|\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text.lower())
		
		nouns = []
		# for np in re_text.split():
		# 	if not np in stopword:
		# 		nouns.append(np)

		doc = nlp(re_text)
		# for np in doc.noun_chunks:
		# 	if not np.lemma_ in stopword:
		# 		nouns.append(np.lemma_)

		words = filter(lambda w : w.pos_ == "NOUN"  and w.lemma_ != "&amp" and 2 < len(re.sub(r'[^A-Za-z0-9_-]+', '', w.lemma_.encode('ascii', errors='ignore')).strip('0123456789_-')),doc)
		for token in words:
			nouns.append(token.lemma_)

		# print 'text= ', text
		# print pos_tag(word_tokenize(text))
		# nouns = [word for (word, pos) in pos_tag(word_tokenize(text)) if is_noun(pos)] 
		i += 1
		if(i%100 == 0):
			print i
		out.append(' '.join(nouns))
	uniq_word = set(' '.join(out).split(' '))
	return out, len(uniq_word)

# if __name__ == '__main__':
	
# 	client = MongoClient()
# 	db = client[db_name]
# 	cursor = db[col_name].find().limit(10000)
	
# 	days = {}
# 	data=1
# 	output = 0
# 	bar = progressbar.ProgressBar(maxval=cursor.count()+1, widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()])
# 	bar.start()
# 	for doc in cursor:
# 		data += 1
# 		bar.update(data)
# 		text = doc['created_at'].split()
# 		location = doc['place']
# 		country = location['country']
# 		if text[4] != '+0000':
# 			print  doc['created_at']
# 		else:
# 			del text[4]
# 			text2 = ' '.join(text)
# 			datetime_object = datetime.datetime.strptime(text2, '%a %b %d %H:%M:%S %Y')
# 			day = get_midnight( datetime_object )
# 			if not days.has_key(day):
# 				days[day] = []
# 			if country not in days[day]:
# 				days[day].append(country)
# 				output += 1
# 			# db[col_name].update({'_id':doc['_id']},{"$set":{'created_day':day}}, upsert=True)
# 	bar.finish()

# 	print(output)

# 	for day, country_list in days.items():
# 		for country in country_list:	
# 			documents = []
# 			cursor = db[col_name].find({'created_day':day, 'place.country':country})
# 			for doc in cursor:
# 				documents.append(doc["text"])

# 			if len(documents) > ignore_under:
# 				print len(documents)
# 				n_doc = preprocess(documents)
# 				nmf_topic,nmf_score = get_NMF(n_doc)
# 				lda_topic, lda_score = get_LDA(n_doc)
# 				db[col_tag_name].update({'day':day, 'country': country},{"$set":{'NMF':nmf_topic,'LDA':lda_topic, 'tweet':len(documents)}}, upsert=True)

# 				for i in range(len(documents)):
# 					db[col_tag_name_2].update({'day':day, 'country': country,'doc':documents[i],'Noun':n_doc[i]},{"$set":{'NMF_topic':nmf_topic[nmf_score[i]],'LDA_topic':lda_topic[lda_score[i]]}}, upsert=True)

st_file = open('stopword.txt')
stopword = st_file.read().split('\n')
st_file.close()

if __name__ == '__main__':
	client = MongoClient()
	db = client[db_name]

	days = []
	if 1==1:
		days = [ datetime.datetime.strptime("2017-10-20T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-10-21T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-10-22T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-10-23T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-10-24T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-10-25T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-10-26T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-10-27T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-10-28T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-10-29T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-02T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-03T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-04T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-05T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-06T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-07T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-08T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-09T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-10T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-11T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-12T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-13T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-16T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-17T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-18T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-19T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-20T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-21T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-22T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-23T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-24T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-25T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-26T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-27T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-28T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-29T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-11-30T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-01T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-02T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-03T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-04T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-05T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-06T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-07T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-08T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-09T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-10T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-11T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-12T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-13T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-14T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-15T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-16T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-17T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-18T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-19T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-20T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-21T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-22T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-29T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-30T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2017-12-31T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2018-01-01T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2018-01-02T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2018-01-03T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2018-01-04T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2018-01-05T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2018-01-06T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2018-01-07T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2018-01-08T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2018-01-09T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2018-01-10T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2018-01-11T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2018-01-12T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2018-01-13T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2018-01-14T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2018-01-15T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
			datetime.datetime.strptime("2018-01-16T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ")]

	# print days[:10]
	for day in days[:3]:
		print day
		documents = []
		cursor = db[col_name].find({'created_day':day, 'place.country':"United States"})
		for doc in cursor:
			documents.append(doc["text"])

		if len(documents) > ignore_under:
			print len(documents)
			n_doc, n_word = preprocess(documents)
			print n_word
			nmf_topic,nmf_score = get_NMF(n_doc)
			lda_topic, lda_score = get_LDA(n_doc)
			db[col_tag_name].update({'day':day, 'country': "United States"},{"$set":{'NMF':nmf_topic,'LDA':lda_topic, 'tweet':len(documents)}}, upsert=True)

			ex_num = min (500,len(documents))
			for i in range(ex_num):
				db[col_tag_name_2].update({'day':day, 'country': "United States",'doc':documents[i]},{"$set":{'Noun':n_doc[i],'NMF_topic':nmf_topic[nmf_score[i]],'LDA_topic':lda_topic[lda_score[i]]}}, upsert=True)




	

