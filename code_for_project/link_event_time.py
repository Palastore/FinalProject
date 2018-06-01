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
col_name = 'event_list'

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

thresholds = 0.7

def get_sim(doc_a,doc_b):
    return 1 - spatial.distance.cosine(doc_a, doc_b)

def get_midnight(time):
    return time.replace(minute=0, hour=0, second=0, microsecond=0)

def get_time_gap(time,hour_gap=1,min_gap=1):
    h = time.hour
    m = time.minute
    o_h = h/hour_gap*hour_gap
    o_m = m/min_gap*min_gap
    return time.replace(hour=o_h, minute=o_m, second=0, microsecond=0)

def read_doc(doc_in_date):
    feature_names = []
    cluster = {}
    for doc in doc_in_date:
        description = doc['description']
        group_id = doc['group_id']
        
        if group_id == -1:
            feature_names = doc['feature_names']
        else:
            if not cluster.has_key(description):
                cluster[description] = {}
            cluster[description][group_id] = {}
            cluster[description][group_id]['centroid'] = doc['centroid']
            cluster[description][group_id]['true_name'] = doc['true_name']
            cluster[description][group_id]['size'] = doc['size']
            
    return feature_names,cluster

def link(time, old_time, description, now_id, link_id):
    db[col_name].update_one({'time':time,'description':description,'group_id':now_id},
                          {"$set":{'past': link_id
                                  }}, upsert=True)
    for g_id in link_id:
        db[col_name].update_one({'time':old_time,'description':description,'group_id':g_id},
                              {"$addToSet":{
                                  'future': now_id
                                  }}, upsert=True)

def union_data(old_key, old_data, new_key, new_data ):
    all_key = list(set().union(old_key,new_key))
    index_old = []
    for key in old_key:
        index_old.append(all_key.index(key))

    index_new = []
    for key in new_key:
        index_new.append(all_key.index(key))

    n = len(all_key)
    old_n = len(old_key)
    new_n = len(new_key)

    all_old_data = []
    all_new_data = []
    for data in old_data:
        buf = [0]*n
        for i in range(old_n):
            buf[ index_old[i] ] = data[i]
        all_old_data.append(buf)

    for data in new_data:
        buf = [0]*n
        for i in range(new_n):
            buf[ index_new[i] ] = data[i]
        all_new_data.append(buf)
    
    return all_key, all_old_data, all_new_data

def link_day(alldoc,time):
    old_time = time - datetime.timedelta(days=1)
    if alldoc.has_key(old_time):
        old_word, old_doc = read_doc(alldoc[old_time])
        now_word, now_doc = read_doc(alldoc[time])
    
        for description in old_doc.keys():
            if now_doc.has_key(description):
                old_n = len(old_doc[description])
                old_centroid = [0]*old_n
                for g_id,data in old_doc[description].items():
                    old_centroid[g_id] = data['centroid']

                now_n = len(now_doc[description])
                now_centroid = [0]*now_n
                for g_id,data in now_doc[description].items():
                    now_centroid[g_id] = data['centroid']
    #             print old_word
    #             print '\n\n'
    #             print now_word
                all_key, all_old_data, all_now_data = union_data(old_word, old_centroid, now_word, now_centroid )
                for i in range(now_n):
                    link_id = []
                    for j in range(old_n):
                        if get_sim(all_now_data[i],all_old_data[j]) > thresholds:
                            link_id.append(j)
                    link(time, old_time, description, i , link_id)
                

if __name__ == '__main__':
	client = MongoClient()
	db = client[db_name]

	result = db[col_name].create_index([('ts', pymongo.ASCENDING)])
	cursor = db[col_name].find({})

	bar = progressbar.ProgressBar(maxval=cursor.count()+1, widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	data=1
	alldoc = {}
	alltime = []
	for doc in cursor:
	    data += 1
	    bar.update(data)
	    date = doc['time']

	    if not alldoc.has_key(date):
	        alldoc[date] = []
	    alldoc[date].append(doc)
	    
	    alltime.append(date)

	alltime = set(alltime)
	    
	bar.finish()

	starttime = datetime.datetime(2012, 6, 1)
	usetime = []
	for day in sorted(alltime):
	    if day >= starttime:
	        usetime.append(day)

	bar = progressbar.ProgressBar(maxval=len(usetime)+1, widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	data=1
	for day in usetime:
	    data += 1
	    bar.update(data)
	    link_day(alldoc,day)
	bar.finish()