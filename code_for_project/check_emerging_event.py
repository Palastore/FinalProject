#! python2
# -*- coding: utf-8 -*-
import pprint
import pymongo
import datetime
import numpy as np
from pymongo import MongoClient

from scipy import spatial

import pylab

import os

import progressbar

db_name = 'twitter'
predict_col_name = 'event_list'
true_col_name = 'after_process_replab'

event_size = [10,15]
author_size = [10,15]

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity

from igraph import *
import igraph
import math
from operator import itemgetter

def get_sim(doc_a,doc_b):
    return 1 - spatial.distance.cosine(doc_a, doc_b)

def get_midnight(time):
    return time.replace(minute=0, hour=0, second=0, microsecond=0)

# Recall = How many relevantitems are selected?
# Precision = How many selecteditems are relevant?
def cal_recall_precision(true, predict):
    recall, precision = 0,0
    if len(true) == 0 and len(predict) == 0:
        recall = 1
        precision = 1
    elif len(true) == 0:
        recall = 0
        precision = 0
    elif len(predict) == 0:
        recall = 0
        precision = 1
    else:
        r = 0
        p = 0
        for x in true:
            if x in predict:
                r += 1
        recall = 1.0*r/len(true)

        for x in predict:
            if x in true:
                p += 1

        precision = 1.0*p/len(predict)
    
    return recall,precision

if __name__ == '__main__':
    client = MongoClient()
    db = client[db_name]

    result = db[predict_col_name].create_index([('ts', pymongo.ASCENDING)])
    cursor = db[predict_col_name].find({})

    bar = progressbar.ProgressBar(maxval=cursor.count()+1, widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    data=1
    predict_event = {}
    predict_event_id = {}
    alltime = []
    for doc in cursor:
        data += 1
        bar.update(data)
        
        date = doc['time']
        description = doc['description']
        group_id = doc['group_id']
        if group_id > 0:
            true_name = doc['true_name']
            tweet_id =doc['tweet_id']
            size = doc['size']
            past = doc['past']
            a_size = doc['author_size']

#             print size,past
            if not predict_event.has_key(date):
                predict_event[date] = {}
                predict_event_id[date] = {}
            if not predict_event[date].has_key(description):
                predict_event[date][description] = {}
                predict_event_id[date][description] = {}
            for i in event_size:
                key = 'e_' + str(i)
                if not predict_event[date][description].has_key(key):
                    predict_event[date][description][key] = []
                    predict_event_id[date][description][key] = []
                
                if size >= i and len(past) == 0:
                    predict_event[date][description][key].append(true_name)
                    predict_event_id[date][description][key].extend(tweet_id)
            
            for i in author_size:
                key = 'a_' + str(i)
                if not predict_event[date][description].has_key(key):
                    predict_event[date][description][key] = []
                    predict_event_id[date][description][key] = []
                
                if a_size >= i and len(past) == 0:
                    predict_event[date][description][key].append(true_name)
                    predict_event_id[date][description][key].extend(tweet_id)
        
        alltime.append(date)
    
    alltime = set(alltime)
        
    bar.finish()

if __name__ == '__main__':
    client = MongoClient()
    db = client[db_name]

    result = db[true_col_name].create_index([('ts', pymongo.ASCENDING)])
    cursor = db[true_col_name].find({'ts':{'$gte':1338508800}}).sort([('ts', pymongo.ASCENDING)])
    
    all_event = {}
    all_event_id = {}
    true_event_id_2 = {}

    bar = progressbar.ProgressBar(maxval=cursor.count()+1, widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    data=1
    for doc in cursor:
        data += 1
        bar.update(data)
        
        ts = doc['ts']
        entity_id = doc['entity_id']
        topic = doc['topic']
        tweet_id = doc['tweet_id']
        topic_dense = doc['topic_dense']
        datetime_object = datetime.datetime.fromtimestamp(ts)
        time_gap = get_midnight(datetime_object)
        
        if not all_event.has_key(time_gap):
            all_event[time_gap] = []
            all_event_id[time_gap] = {}
            true_event_id_2[time_gap] = []
        true_event_id_2[time_gap].append(tweet_id)
#         if tweet_id =='208604422492786688':
#             print 'in',time_gap
        if topic_dense == 1:
#         if True:
            if not (topic in all_event[time_gap]):
                 all_event[time_gap].append(topic)
            if not (all_event_id[time_gap].has_key(topic)):
                all_event_id[time_gap][topic] = []
            all_event_id[time_gap][topic].append(tweet_id)

        
    bar.finish()
#     print len(date)

starttime = datetime.datetime(2012, 6, 1)
usetime = []
true_event = {}
true_event_id = {}


for day in sorted(alltime):
    if day >= starttime:
        usetime.append(day)
        true_event[day] = []
        true_event_id[day] = []

for now_time, event_list in all_event.items():
    old_time = now_time - datetime.timedelta(days=1)
    for event in event_list:
        if all_event.has_key(old_time):
            if not (event in all_event[old_time]):
                if  true_event.has_key(now_time):
                    true_event[now_time].append(event)
                    true_event_id[now_time].extend( all_event_id[now_time][event] )
        else:
            if  true_event.has_key(now_time):
                true_event[now_time].append(event)
                true_event_id[now_time].extend( all_event_id[now_time][event] )

all_score = {}
all_score_id = {}
bar = progressbar.ProgressBar(maxval=len(usetime)+1, widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
data=1
for day in usetime:
    data += 1
    bar.update(data)
    for description,d_list in predict_event[day].items():
        if not all_score.has_key(description):
            all_score[description] = {}
            all_score_id[description] = {}

        for key,d_event in d_list.items():
            if not all_score[description].has_key(key):
                all_score[description][key] = {}
                all_score[description][key]['recall'] = []
                all_score[description][key]['precision'] = []
            
                all_score_id[description][key] = {}
                all_score_id[description][key]['recall'] = []
                all_score_id[description][key]['precision'] = []
            recall, precision = cal_recall_precision(true_event[day], d_event)
            all_score[description][key]['recall'].append(recall)
            all_score[description][key]['precision'].append(precision)

            recall, precision = cal_recall_precision(true_event_id[day], predict_event_id[day][description][key])
            all_score_id[description][key]['recall'].append(recall)
            all_score_id[description][key]['precision'].append(precision)
bar.finish()

for description,list_data in all_score.items():
    for type_key, score_data in list_data.items():
        for score_key,score in score_data.items():
            print description, type_key,score_key, sum(score)/len(score)
        print ''