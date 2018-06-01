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
col_name = 'cluster_score'

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

def get_thai_midnight(time):
    out = time + datetime.timedelta(hours=7)
    out = out.replace(minute=0, hour=0, second=0, microsecond=0) - datetime.timedelta(hours=7)
    return out

if __name__ == '__main__':
	client = MongoClient()
	db = client[db_name]

	result = db[col_name].create_index([('ts', pymongo.ASCENDING)])
	cursor = db[col_name].find({})

	bar = progressbar.ProgressBar(maxval=cursor.count()+1, widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	data=1
	all_score = {}
	d_all_score = {}
	for doc in cursor:
	    data += 1
	    bar.update(data)
	    description = doc['description']
	    levels = doc['levels']
	    score_type = doc['type']
	    
	    completeness_score = doc['completeness_score']
	    homogeneity_score = doc['homogeneity_score']
	    v_measure_score = doc['v_measure_score']
	    accuracy = doc['accuracy']

	    if not all_score.has_key(description):
	        all_score[description] = {}
	        d_all_score[description] = {}
	        d_all_score[description]['completeness_score'] = []
	        d_all_score[description]['homogeneity_score'] = []
	        d_all_score[description]['v_measure_score'] = []
	        d_all_score[description]['accuracy'] = []
	    
	    if not all_score[description].has_key(levels):
	        all_score[description][levels] = {}
	        
	    if not all_score[description][levels].has_key(score_type):
	        all_score[description][levels][score_type] = {}
	        all_score[description][levels][score_type]['completeness_score'] = []
	        all_score[description][levels][score_type]['homogeneity_score'] = []
	        all_score[description][levels][score_type]['v_measure_score'] = []
	        all_score[description][levels][score_type]['accuracy'] = []

	        
	    all_score[description][levels][score_type]['completeness_score'].append(completeness_score)
	    all_score[description][levels][score_type]['homogeneity_score'].append(homogeneity_score)
	    all_score[description][levels][score_type]['v_measure_score'].append(v_measure_score)
	    all_score[description][levels][score_type]['accuracy'].append(accuracy)
	    
	    d_all_score[description]['completeness_score'].append(completeness_score)
	    d_all_score[description]['homogeneity_score'].append(homogeneity_score)
	    d_all_score[description]['v_measure_score'].append(v_measure_score)
	    d_all_score[description]['accuracy'].append(accuracy)
	    
	      
	bar.finish()

	for description,v1 in all_score.items():
	    for levels,v2 in v1.items():
	        for score_type,v3 in v2.items():
	            for score_name, score_list in v3.items():
	                print description,levels,score_type,score_name, 1.0*sum(score_list)/len(score_list)
	            print ''

	for description,v1 in d_all_score.items():
	    for score_name, score_list in v1.items():
	        print description,score_name, 1.0*sum(score_list)/len(score_list)
	    print ''