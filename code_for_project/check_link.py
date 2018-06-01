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
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity

from igraph import *
import igraph
import math
from operator import itemgetter

thresholds = 0.5

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
        

        if group_id >=0 :
            if not cluster.has_key(description):
                cluster[description] = {}
            cluster[description][group_id] = {}
            cluster[description][group_id]['centroid'] = doc['centroid']
            cluster[description][group_id]['true_name'] = doc['true_name']
            cluster[description][group_id]['size'] = doc['size']
            cluster[description][group_id]['past'] = doc['past']
            cluster[description][group_id]['future'] = doc['future']
            
    return cluster

def cal_matrix_score(matrix_use, true_name, predict):
    labels_true = []
    labels = []
    
    labels_true_use = []
    labels_use = []
    
    x_n = len(matrix_use)
#     c = 0
#     e = 0
    for i in range(x_n):
        y_n = len(matrix_use[i])
        for j in range(y_n):
            labels_true.append(true_name[i][j])
            labels.append(predict[i][j])
            if matrix_use[i][j] == 1:
                labels_true_use.append(true_name[i][j])
                labels_use.append(predict[i][j])
#                 if true_name[i][j] !=  predict[i][j]:
#                     print 'BUG???',true_name[i][j],predict[i][j]
#                     e += 1
#             c += 1
#     print e,'/',c
#     print labels_true_use
#     print labels_use
    score = {}
    acc_score = accuracy_score(labels_true, labels)
    acc_score_use = accuracy_score(labels_true_use, labels_use)
    score['accuracy_score'] = acc_score
    return score

def check_link(alldoc,time):
    old_time = time - datetime.timedelta(days=1)
    des_score = {}
    if alldoc.has_key(old_time):
        old_doc = read_doc(alldoc[old_time])
        now_doc = read_doc(alldoc[time])
    
        for description in old_doc.keys():
            if now_doc.has_key(description):
                old_n = len(old_doc[description])
                old_name = [0]*old_n
                old_size = [0]*old_n
                for o_id,data in old_doc[description].items():
                    old_name[o_id] = data['true_name']
                    old_size[o_id] = data['size']
                    
                now_n = len(now_doc[description])
                now_name = [0]*now_n
                now_size = [0]*now_n
                
                predict = [[0 for x in range(old_n)] for y in range(now_n)] 
                true_name = [[0 for x in range(old_n)] for y in range(now_n)]
                matrix_use = [[0 for x in range(old_n)] for y in range(now_n)]
                
                for n_id,data in now_doc[description].items():
                    now_name[n_id] = data['true_name']
                    now_size[n_id] = data['size']
                    link = data['past']
                    for o_id in link:
                        predict[n_id][o_id] = 1
                    
                for n_id in range(now_n):
                    for o_id in range(old_n):
                        if old_name[o_id] == now_name[n_id]:
                            true_name[n_id][o_id] = 1
                        if old_size[o_id] >= 3 and now_size[n_id] >= 3:
                            matrix_use[n_id][o_id] = 1
                
                des_score[description] = cal_matrix_score(matrix_use, true_name, predict)
    return des_score

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
	    
	all_score = {}
	bar = progressbar.ProgressBar(maxval=len(usetime)+1, widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	data=1
	for day in usetime:
	    data += 1
	    bar.update(data)
	    des_score = check_link(alldoc,day)
	    for description,score_data in des_score.items():
	        if not all_score.has_key(description):
	            all_score[description] = {}
	        for score_key,score in score_data.items():
	            if not all_score[description].has_key(score_key):
	                all_score[description][score_key] = []
	            all_score[description][score_key].append(score)
	bar.finish()

	for description,score_data in all_score.items():
	    for score_key,score in score_data.items():
	        print description, score_key, sum(score)/len(score)
	    print ''