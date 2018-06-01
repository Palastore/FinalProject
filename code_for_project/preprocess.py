#! python2
# -*- coding: utf-8 -*-
import pprint
import pymongo
import datetime
from pymongo import MongoClient

import os
import re

from nltk import word_tokenize, pos_tag
from nltk.tokenize import RegexpTokenizer 
from nltk.stem.porter import PorterStemmer 

import progressbar

import spacy

nlp = spacy.load('en_core_web_sm') 

db_name = 'twitter'
col_name = 'replab'

col_tag_name = 'after_process_replab'

col_update = 'update_time'

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

emoji_pattern = re.compile(
		u"(\ud83d[\ude00-\ude4f])|"  # emoticons
		u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
		u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
		u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
		u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
		"+", flags=re.UNICODE)

def preprocess(text):
	hashtag = []
	for tag in text.split():
		if tag.startswith("#"):
			hashtag.append(tag)
	mention = []
	for men in text.split():
		if men.startswith("@"):
			mention.append(men)
	re_text = re.sub(r'(#[^\s]+)|(@[^\s]+)|\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text.lower())
	re_text = emoji_pattern.sub(r'', re_text)

	nouns = []

	doc = nlp(re_text)

	words = filter(lambda w : w.pos_ == "NOUN"  and w.lemma_ != "&amp" and 2 < len(re.sub(r'[^A-Za-z0-9_-]+', '', w.lemma_.encode('ascii', errors='ignore')).strip('0123456789_-')),doc)
	for token in words:
		if not ((token.lemma_ in stopword ) or (token.text in stopword)):
			nouns.append(token.lemma_)

	return nouns,hashtag,mention

tokenizer = RegexpTokenizer(r'\w+') 
p_stemmer = PorterStemmer() 
is_noun = lambda pos: pos[:2] in ['NN', 'NNS', 'NNP', 'NNPS'] 

def preprocess_nltk(text):
	hashtag = []
	for tag in text.split():
		if tag.startswith("#"):
			hashtag.append(tag)
	mention = []
	for men in text.split():
		if men.startswith("@"):
			mention.append(men)
	raw = re.sub(r'(#[^\s]+)|(@[^\s]+)|\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text.lower())
	re_text = emoji_pattern.sub(r'', raw)
	
	# clean and tokenize document string 
	# print 'raw = ',raw 
	tokens = re_text.split(' ')
	# print 'tokens = ',tokens 

	# remove stop words from tokens 
	stopped_tokens = [i for i in tokens if not i in stopword] 
	# print 'stopped_tokens = ',stopped_tokens 

	tokens = tokenizer.tokenize(' '.join(stopped_tokens) ) 

	stopped_tokens = [i for i in tokens if not i in stopword] 

	nouns = [word for (word, pos) in pos_tag(stopped_tokens) if is_noun(pos)] 

	# stem tokens 
	stemmed_tokens = [p_stemmer.stem(i) for i in nouns] 
	# print 'stemmed_tokens = ',stemmed_tokens 

	return stemmed_tokens,hashtag,mention


if __name__ == '__main__':
	
	client = MongoClient()
	db = client[db_name]
	
	result = db[col_name].create_index([('ts', pymongo.ASCENDING)])
	cursor = db[col_update].find_one({'tag':'preprocess_replab'}) 

	st_file = open('stopword.txt')
	stopword = st_file.read().split('\n')
	st_file.close()

	if not cursor:
		cursor = db[col_name].find().sort([('ts', pymongo.ASCENDING)])

	else:
		col_update_time = cursor['ts']
		cursor = db[col_name].find({'ts': { "$gte": col_update_time }}).sort([('ts', pymongo.ASCENDING)])

	bar = progressbar.ProgressBar(maxval=cursor.count()+1, widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	data=1
	for doc in cursor:
		data += 1
		bar.update(data)

		ts = doc['ts']
		datetime_object = datetime.datetime.fromtimestamp(ts/1000)
		day = get_midnight( datetime_object )

		nouns,hashtag,mention = preprocess(doc['text'])
		nouns_nltk,hashtag,mention = preprocess_nltk(doc['text'])
			# db[col_name].update({'_id':doc['_id']},{"$set":{'created_day':day}}, upsert=True)

		if not db[col_tag_name].find_one({'_id':doc['_id']}):
			db[col_tag_name].insert(doc)
		db[col_tag_name].update({'_id':doc['_id']},{"$set":{'created_date':datetime_object,'created_day':day, 'nouns':nouns, 'nouns_nltk':nouns_nltk,'hashtags':hashtag,'mentions':mention}}, upsert=True)
		db[col_update].update({'tag':'preprocess_replab'},{"$set":{'ts':ts}}, upsert=True)

	bar.finish()
	