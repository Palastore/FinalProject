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
col_name = 'after_process_replab'

event_size = 10

from sklearn import metrics
from sklearn.cluster import DBSCAN
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity

from igraph import *
import igraph
import math
from operator import itemgetter

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

def get_time_gap(time,hour_gap=1,min_gap=1):
    h = time.hour
    m = time.minute
    o_h = h/hour_gap*hour_gap
    o_m = m/min_gap*min_gap
    return time.replace(hour=o_h, minute=o_m, second=0, microsecond=0)

def get_week_year(time):
    return tuple([time.isocalendar()[0], time.isocalendar()[1]])

topic_priority = [ "MILDLY_IMPORTANT", "", "UNIMPORTANT", "ALERT", "NEUTRAL" ]
focus_priority = [ "MILDLY_IMPORTANT", "ALERT"]
# focus_priority = [ "ALERT"]

f = open("replab2013_entities.tsv","r") #opens file with name of "test.txt"
class_set = {}
for line in f:
    x = line.split('\t')
#     class_set.append( line)
    class_set[ x[0].strip('"') ] = [x[1].strip('"'),x[3].strip('"')] 
# print class_set

if __name__ == '__main__':
    client = MongoClient()
    db = client[db_name]

    result = db[col_name].create_index([('ts', pymongo.ASCENDING)])
    cursor = db[col_name].find({})

    bar = progressbar.ProgressBar(maxval=cursor.count()+1, widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    data=1
    topic_per_time = {}
    for doc in cursor:
        data += 1
        bar.update(data)
        ts = doc['ts']
        entity_id = doc['entity_id']
        topic = doc['topic']
       
        minor_class = class_set[entity_id][0]
        
        major_class = class_set[entity_id][1]
        
        db[col_name].update_one({'_id':doc['_id']},{"$set":{'major_class':major_class,'minor_class':minor_class}}, upsert=True)
        
        
        datetime_object = datetime.datetime.fromtimestamp(ts)
        time_gap = get_midnight(datetime_object)
        
        if not topic_per_time.has_key(time_gap):
            topic_per_time[time_gap] = {}
        if not topic_per_time[time_gap].has_key(topic):
             topic_per_time[time_gap][topic] = 0
        topic_per_time[time_gap][topic] += 1
          
    bar.finish()
    
    ignore_topics = ['picture','link']
    
    cursor = db[col_name].find({})

    bar = progressbar.ProgressBar(maxval=cursor.count()+1, widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    data=1
    for doc in cursor:
        data += 1
        bar.update(data)
        ts = doc['ts']
        entity_id = doc['entity_id']
        topic = doc['topic']
        priority = doc['topic_priority']
        
        datetime_object = datetime.datetime.fromtimestamp(ts)
        time_gap = get_midnight(datetime_object)
        
        topic_dense = 0
        topic_l = topic.lower()
        check = True
        if topic =='' or priority == '':
            check = False
        for ignore in ignore_topics:
            if ignore in topic_l:
                check = False
        if not check:
            topic_dense = -1
#         if topic_per_time[time_gap][topic] >= event_size and check:
#             topic_dense = 1
        if priority in focus_priority:
            topic_dense = 1
        db[col_name].update_one({'_id':doc['_id']},{"$set":{'topic_dense':topic_dense}}, upsert=True)
    bar.finish()
#     print len(date)