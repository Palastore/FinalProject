#! python2
# -*- coding: utf-8 -*-
import pprint
import pymongo
import datetime
import numpy as np
from pymongo import MongoClient

from scipy import spatial
from scipy.sparse import isspmatrix, dok_matrix, csc_matrix
import sklearn.preprocessing

import pylab

import os

import progressbar

db_name = 'twitter'
col_name = 'after_process_replab'

col_output_score = 'cluster_score'
col_output_event = 'event_list'

from sklearn import metrics
from sklearn.cluster import DBSCAN, AffinityPropagation, MeanShift, estimate_bandwidth
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances

from igraph import *
import igraph
import math
from operator import itemgetter

def get_midnight(time):
    return time.replace(minute=0, hour=0, second=0, microsecond=0)

def get_time_gap(time,hour_gap=1,min_gap=1):
    h = time.hour
    m = time.minute
    o_h = h/hour_gap*hour_gap
    o_m = m/min_gap*min_gap
    return time.replace(hour=o_h, minute=o_m, second=0, microsecond=0)

def get_sim(doc_a,doc_b):
    return 1 - spatial.distance.cosine(doc_a, doc_b)

def view_group(n_cluster, all_doc, doc_key, g_id):
        group = []
        for i in range(n_cluster):
            group.append([])
        
        for i in range( len(g_id) ):
            group[ g_id[i] ].append(all_doc[i][doc_key])
        return group

def split_day(date,text,all_doc):
    s_text = {}
    s_all_doc = {}
    for datetime_object in date:
        i = date.index(datetime_object)
        key = get_midnight(datetime_object)
        if not s_text.has_key(key):
            s_text[key] = []
            s_all_doc[key] = []
        s_text[key].append(text[i])
        s_all_doc[key].append(all_doc[i])
    return s_text,s_all_doc

def dbscan(test):
	# Compute DBSCAN
	cluster = DBSCAN(eps=0.3, min_samples=3,metric='precomputed').fit(test)
	# distance <= eps
	core_samples_mask = np.zeros_like(cluster.labels_, dtype=bool)
	core_samples_mask[cluster.core_sample_indices_] = True
	labels = cluster.labels_

	#     print cluster.core_sample_indices_

	#     print labels
	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	#     print('Estimated number of clusters: %d' % n_clusters_)
	#     if n_clusters_>1:
	#         print("Silhouette Coefficient: %0.3f"
	#           % metrics.silhouette_score(test, labels))
	    
	group = []
	for i in range(n_clusters_+1):
	    group.append([])
	index = 0
	for i in labels:
	    group[i].append(index)
	    index += 1
	    
	re_labels = -1
	for i in range(len(labels)):
	    if labels[i] == -1:
	        labels[i] = re_labels
	        re_labels -= 1

	group = group[:-1]
	# for i in range(n_clusters_):
	#     print len(group[i])

	return group, labels, n_clusters_


#AffinityPropagation
def aff_cluster(test):
	cluster = AffinityPropagation(affinity='euclidean').fit(test)
	cluster_centers = cluster.cluster_centers_indices_
	labels = cluster.labels_
	#     print cluster_centers_indices
	#     print labels

	# print labels
	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels))
	#     print('Estimated number of clusters: %d' % n_clusters_)
	    
	group = []
	for i in range(n_clusters_):
	    group.append([])
	index = 0
	for i in labels:
	    group[i].append(index)
	    index += 1
	    
	regroup = []
	relabel = []
	recenters = []
	index = 0
	for mem in group:
	    if len(mem) >= 3:
	        regroup.append(mem)
	        relabel.append(index)
	        recenters.append(cluster_centers[index])
	#             print index
	    index += 1

	new_label = []
	for i in labels:
	    if i in relabel:
	        new_label.append(relabel.index(i))
	    else:
	        new_label.append(-1)
	# for i in range(n_clusters_):
	#     print len(group[i])
	#     print new_label
	new_n_clusters_ = len(set(new_label)) - (1 if -1 in labels else 0)
	#     print('Estimated number of clusters: %d' % n_clusters_)

	return group, labels, n_clusters_, cluster_centers, regroup, new_label, new_n_clusters_, recenters

# import markov_clustering as mc
# py 2 not have
def sparse_allclose(a, b, rtol=1e-5, atol=1e-8):
    """
    Version of np.allclose for use with sparse matrices
    """
    c = np.abs(a - b) - rtol * np.abs(b)
    # noinspection PyUnresolvedReferences
    return c.max() <= atol


def normalize(matrix):
    """
    Normalize the columns of the given matrix
    
    :param matrix: The matrix to be normalized
    :returns: The normalized matrix
    """
    return sklearn.preprocessing.normalize(matrix, norm="l1", axis=0)


def inflate(matrix, power):
    """
    Apply cluster inflation to the given matrix by raising
    each element to the given power.
    
    :param matrix: The matrix to be inflated
    :param power: Cluster inflation parameter
    :returns: The inflated matrix
    """
    if isspmatrix(matrix):
        return normalize(matrix.power(power))

    return normalize(np.power(matrix, power))


def expand(matrix, power):
    """
    Apply cluster expansion to the given matrix by raising
    the matrix to the given power.
    
    :param matrix: The matrix to be expanded
    :param power: Cluster expansion parameter
    :returns: The expanded matrix
    """
    if isspmatrix(matrix):
        return matrix ** power

    return np.linalg.matrix_power(matrix, power)


def add_self_loops(matrix, loop_value):
    """
    Add self-loops to the matrix by setting the diagonal
    to loop_value
    
    :param matrix: The matrix to add loops to
    :param loop_value: Value to use for self-loops
    :returns: The matrix with self-loops
    """
    shape = matrix.shape
    assert shape[0] == shape[1], "Error, matrix is not square"

    if isspmatrix(matrix):
        new_matrix = matrix.todok()
    else:
        new_matrix = matrix.copy()

    for i in range(shape[0]):
        new_matrix[i, i] = loop_value

    if isspmatrix(matrix):
        return new_matrix.tocsc()

    return new_matrix


def prune(matrix, threshold):
    """
    Prune the matrix so that very small edges are removed
    
    :param matrix: The matrix to be pruned
    :param threshold: The value below which edges will be removed
    :returns: The pruned matrix
    """
    if isspmatrix(matrix):
        pruned = dok_matrix(matrix.shape)
        pruned[matrix >= threshold] = matrix[matrix >= threshold]
        pruned = pruned.tocsc()
    else:
        pruned = matrix.copy()
        pruned[pruned < threshold] = 0

    return pruned


def converged(matrix1, matrix2):
    """
    Check for convergence by determining if 
    matrix1 and matrix2 are approximately equal.
    
    :param matrix1: The matrix to compare with matrix2
    :param matrix2: The matrix to compare with matrix1
    :returns: True if matrix1 and matrix2 approximately equal
    """
    if isspmatrix(matrix1) or isspmatrix(matrix2):
        return sparse_allclose(matrix1, matrix2)

    return np.allclose(matrix1, matrix2)


def iterate(matrix, expansion, inflation):
    """
    Run a single iteration (expansion + inflation) of the mcl algorithm
    
    :param matrix: The matrix to perform the iteration on
    :param expansion: Cluster expansion factor
    :param inflation: Cluster inflation factor
    """
    # Expansion
    matrix = expand(matrix, expansion)

    # Inflation
    matrix = inflate(matrix, inflation)

    return matrix


def get_clusters(matrix):
    """
    Retrieve the clusters from the matrix
    
    :param matrix: The matrix produced by the MCL algorithm
    :returns: A list of tuples where each tuple represents a cluster and
              contains the indices of the nodes belonging to the cluster
    """
    if not isspmatrix(matrix):
        # cast to sparse so that we don't need to handle different 
        # matrix types
        matrix = csc_matrix(matrix)

    # get the attractors - non-zero elements of the matrix diagonal
    attractors = matrix.diagonal().nonzero()[0]

    # somewhere to put the clusters
    clusters = set()

    # the nodes in the same row as each attractor form a cluster
    for attractor in attractors:
        cluster = tuple(matrix.getrow(attractor).nonzero()[1].tolist())
        clusters.add(cluster)

    return sorted(list(clusters))


def run_mcl(matrix, expansion=2, inflation=2, loop_value=1,
				iterations=100, pruning_threshold=0.001, pruning_frequency=1,
				convergence_check_frequency=1, verbose=False):
	"""
	Perform MCL on the given similarity matrix

	:param matrix: The similarity matrix to cluster
	:param expansion: The cluster expansion factor
	:param inflation: The cluster inflation factor
	:param loop_value: Initialization value for self-loops
	:param iterations: Maximum number of iterations
	       (actual number of iterations will be less if convergence is reached)
	:param pruning_threshold: Threshold below which matrix elements will be set
	       set to 0
	:param pruning_frequency: Perform pruning every 'pruning_frequency'
	       iterations. 
	:param convergence_check_frequency: Perform the check for convergence
	       every convergence_check_frequency iterations
	:param verbose: Print extra information to the console
	:returns: The final matrix
	"""
	#     print("-" * 50)
	#     print("MCL Parameters")
	#     print("Expansion: {}".format(expansion))
	#     print("Inflation: {}".format(inflation))
	#     if pruning_threshold > 0:
	#         print("Pruning threshold: {}, frequency: {} iteration{}".format(
	#             pruning_threshold, pruning_frequency, "s" if pruning_frequency > 1 else ""))
	#     else:
	#         print("No pruning")
	#     print("Convergence check: {} iteration{}".format(
	#         convergence_check_frequency, "s" if convergence_check_frequency > 1 else ""))
	#     print("Maximum iterations: {}".format(iterations))
	#     print("{} matrix mode".format("Sparse" if isspmatrix(matrix) else "Dense"))
	#     print("-" * 50)

	# Initialize self-loops
	if loop_value > 0:
	    matrix = add_self_loops(matrix, loop_value)

	# Normalize
	matrix = normalize(matrix)

	# iterations
	for i in range(iterations):
	#         print("Iteration {}".format(i + 1))

	    # store current matrix for convergence checking
	    last_mat = matrix.copy()

	    # perform MCL expansion and inflation
	    matrix = iterate(matrix, expansion, inflation)

	    # prune
	    if pruning_threshold > 0 and i % pruning_frequency == pruning_frequency - 1:
	#             print("Pruning")
	        matrix = prune(matrix, pruning_threshold)

	    # Check for convergence
	    if i % convergence_check_frequency == convergence_check_frequency - 1:
	#             print("Checking for convergence")
	        if converged(matrix, last_mat):
	#                 print("Converged after {} iteration{}".format(i + 1, "s" if i > 0 else ""))
	            break

	#     print("-" * 50)

	return matrix

# for py 3.xx 
# import markov_clustering as mc

# mcl = mc.run_mcl(test,inflation = 1.5)           # run MCL with default parameters
# cluster = mc.get_clusters(mcl) 

def label_mcl(cluster, n):
    labels = [0]*n
    filter_labels = [0]*n
    i = 0
    filter_i = -1
    num_filter = 0
    filter_cluster = []
    for mem in cluster:
        if len(mem) <3:
            num_filter = -1
        else:
            filter_i += 1
            num_filter = filter_i
            filter_cluster.append(mem)
        for index in mem:
            labels[index] = i
            filter_labels[index] = num_filter
        i += 1
    return labels, filter_labels,filter_cluster 

def get_labels_true(all_doc):
    topic_labels = []
    minor_class_labels = []
    major_class_labels = []
    topic_dense = []
    for doc in all_doc:
        topic_labels.append(doc['topic'])
        minor_class_labels.append(doc['minor_class'])
        major_class_labels.append(doc['major_class'])
        topic_dense.append(doc['topic_dense'])
    return topic_labels,minor_class_labels,major_class_labels,topic_dense

def rend_index(labels_true, labels):
    tp_tn = 0 #same cluster in both labels + diff cluster in both labels
    n = len(labels_true)
    for i in range(n):
        for j in range(i+1,n):
            lt_same = (labels_true[i] == labels_true[j])
            l_same = (labels[i] == labels[j])
            if lt_same == l_same:
                tp_tn += 1
    return 1.0*tp_tn/(n*(n-1)/2)

def cal_cluster_score(labels_true, labels):
    homogeneity, completeness, v_measure  = metrics.homogeneity_completeness_v_measure(labels_true, labels)
    acc = rend_index(labels_true, labels)
    return homogeneity, completeness, v_measure, acc
#     adjusted_rand_score =  metrics.adjusted_rand_score(labels_true, labels)
#     adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(labels_true, labels)
#     fowlkes_mallows_score = metrics.fowlkes_mallows_score(labels_true, labels)
#     return homogeneity, completeness, v_measure, adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score


# Measured in three levels. major_class > minor_class > topic


# 1 : use all topic 
# 2 : use only topic happen > 3 time and not ignore topic in true label and topic that Selected by cluster algorithm 

def cluster_score(time,levels,description,labels_true, labels,topic_dense):
	n = len(labels_true)
	#1
	homogeneity, completeness, v_measure, acc = cal_cluster_score(labels_true, labels)

	db[col_output_score].update_one({'time':time,'levels':levels,'description':description,'type':1},
	                          {"$set":{'homogeneity_score':homogeneity, 
	                                   'completeness_score':completeness, 
	                                   'v_measure_score':v_measure, 
	                                   'accuracy':acc,
	#                                        'adjusted_rand_score':adjusted_rand_score, 
	#                                        'adjusted_mutual_info_score':adjusted_mutual_info_score, 
	#                                        'fowlkes_mallows_score':fowlkes_mallows_score
	                                  }}, upsert=True)

	#2
	filter_labels_true = []
	filter_labels = []
	for i in range(n):
	    if topic_dense[i] == 1 or labels[i] >= 0:
	        filter_labels_true.append(labels_true[i])
	        filter_labels.append(labels[i])
	homogeneity, completeness, v_measure, acc = cal_cluster_score(filter_labels_true, filter_labels)
	db[col_output_score].update_one({'time':time,'levels':levels,'description':description,'type':2},
	                  {"$set":{'homogeneity_score':homogeneity, 
	                           'completeness_score':completeness, 
	                           'v_measure_score':v_measure, 
	                           'accuracy':acc,
	#                                'adjusted_rand_score':adjusted_rand_score, 
	#                                'adjusted_mutual_info_score':adjusted_mutual_info_score, 
	#                                'fowlkes_mallows_score':fowlkes_mallows_score
	                          }}, upsert=True)

def find_centroid_score(tf,group_member,feature_names,topic_name):
    centroid_list = []
    centroid_word_list = []
    score_list = []
    centroid_name_list = []
    size_list = []
    
    for group in group_member:
        size_list.append(len(group))
        most_sim = -1
        most_sim_index = 0
        count = np.zeros(len(feature_names))
        for member in group:
            count += tf[member].toarray()[0]
        centroid = count / len(group)
        centroid_list.append(centroid)
        
        centroid_word = []
        for word_index in range(len(feature_names)):
            if count[word_index] != 0:
                word_data = tuple([centroid[word_index] ,feature_names[word_index]])
                centroid_word.append(word_data)
        centroid_word_list.append( sorted(centroid_word, reverse  = True)[:5])
            
        sum_sim = 0
        for member in group:
            m_tf = tf[member].toarray()[0]
            sim = get_sim(m_tf,centroid)
            sum_sim += sim
            
            if sim > most_sim:
                most_sim_index = member
                most_sim = sim
        score = sum_sim/len(group)
        score_list.append(score)
        
        centroid_name_list.append(topic_name[most_sim_index])
        
        
    return centroid_list, centroid_word_list, score_list, centroid_name_list, size_list

def find_centroid_score_aff(tf,group_member,feature_names,topic_name,centroid_list_index):
    centroid_list = []
    centroid_word_list = []
    score_list = []
    centroid_name_list = []
    size_list = []
    i = 0
    for group in group_member:
        size_list.append(len(group))
        centroid_word = []
        centroid = np.zeros(len(feature_names)) + tf[centroid_list_index[i]].toarray()[0]
        centroid_list.append(centroid)
        
        for word_index in range(len(feature_names)):
            if centroid[word_index] != 0:
                word_data = tuple([centroid[word_index] ,feature_names[word_index]])
                centroid_word.append(word_data)
        centroid_word_list.append( sorted(centroid_word, reverse  = True)[:5])
            
        sum_sim = 0
        for member in group:
            m_tf = tf[member].toarray()[0]
            sim = get_sim(m_tf,centroid)
            sum_sim += sim
        
        score = sum_sim/len(group)
        score_list.append(score)
        
        centroid_name_list.append(topic_name[ centroid_list_index[i] ])
        
        i +=1
        
        
    return centroid_list, centroid_word_list, score_list, centroid_name_list, size_list

def write_event(time,description, feature_names, centroid_list, centroid_word_list, score_list, centroid_name_list, size_list,tweet_id_list, author_list):
    n = len(centroid_list)
    db[col_output_event].update_one({'time':time,'description':'feature_names','group_id':-1},
                          {"$set":{'feature_names':feature_names, 
                                  }}, upsert=True)
    for i in range(n):
        centroid = tuple(centroid_list[i])
        name = centroid_word_list[i]
        score = score_list[i]
        true_name = centroid_name_list[i]
        size = size_list[i]
        tweet_id = tweet_id_list[i]
        author =  author_list[i]
        author_size = len(author)
        db[col_output_event].update_one({'time':time,'description':description,'group_id':i},
                          {"$set":{'centroid': centroid, 
                                   'name':name, 
                                   'score': score,
                                   'true_name': true_name,
                                   'size': size,
                                   'tweet_id':tweet_id,
                                   'author' :author,
                                   'author_size' :author_size,
                                   'past': [],
                                   'future': [],
                                  }}, upsert=True)
    
def get_tweet_id(sample_doc, group):
    tweet_id_list = []
    author_list = []
    for member in group:
        buf = []
        buf2 = set()
        for i in member:
            tweet_id = sample_doc[i]['tweet_id']
            author = sample_doc[i]['author']
            buf.append(tweet_id)
            buf2.add(author)
        tweet_id_list.append(buf)
        author_list.append(list(buf2))
    return tweet_id_list, author_list  

if __name__ == '__main__':
	client = MongoClient()
	db = client[db_name]

	result = db[col_name].create_index([('ts', pymongo.ASCENDING)])
	cursor = db[col_name].find({ 'ts':{'$gte':1338508800},'topic_dense':{'$in':[0,1]} }).sort([('ts', pymongo.ASCENDING)])

	date = []
	text = []
	all_doc = []

	bar = progressbar.ProgressBar(maxval=cursor.count()+1, widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	data=1
	for doc in cursor:
	    data += 1
	    bar.update(data)
	    
	    ts = doc['ts']
	    datetime_object = datetime.datetime.fromtimestamp(ts)
	    if True:
	        buf = doc['nouns_nltk']
	        for x in doc['hashtags']:
	            if not x.lower() in buf:
	                buf.append(x.lower())
	        all_doc.append(doc)
	        date.append(datetime_object)
	        text.append(' '.join(buf))
	bar.finish()
	#     print len(date)

	s_text,s_all_doc = split_day(date,text,all_doc)

	bar = progressbar.ProgressBar(maxval=len(s_text)+1, widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	data=1

	for date in s_text.keys():
	    data += 1
	    bar.update(data)
	    
	    sample = s_text[date]
	    sample_doc = s_all_doc[date]
	    
	    if(len(sample) < 20):
	        continue
	    
	    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2)
	    tf = tf_vectorizer.fit_transform(sample)
	    feature_names = tf_vectorizer.get_feature_names()
	    
	    cosine_sim = cosine_similarity(tf)
	    cosine_dis = cosine_distances(tf)
	    
	    #  get labels_true 
	    topic_labels, minor_class_labels, major_class_labels, topic_dense = get_labels_true(sample_doc)
	    
	    # DBSCAN  
	    db_group, db_labels, db_n_clusters = dbscan(cosine_dis)
	    
	    centroid_list, centroid_word_list, score_list, centroid_name_list, size_list = find_centroid_score(tf,db_group,feature_names,topic_labels)
	#     print type(centroid_list),centroid_list
	    tweet_id_list, author_list = get_tweet_id(sample_doc, db_group)
	    write_event(date,'DBSCAN', feature_names, centroid_list, centroid_word_list, score_list, centroid_name_list, size_list, tweet_id_list, author_list)
	    
	    
	    # Affinity Propagation  
	    aff_group, aff_labels, aff_n_clusters, aff_cluster_centers, aff_filter_group, aff_filter_labels, aff_filter_n_clusters, aff_filter_centers = aff_cluster(tf)
	    
	    centroid_list, centroid_word_list, score_list, centroid_name_list, size_list = find_centroid_score_aff(tf, aff_group, feature_names, topic_labels, aff_cluster_centers)
	#     print type(centroid_list),centroid_list
	    tweet_id_list, author_list = get_tweet_id(sample_doc, aff_group)
	    
	    write_event(date,'Affinity Propagation', feature_names, centroid_list, centroid_word_list, score_list, centroid_name_list, size_list, tweet_id_list, author_list)
	     
	    
	    # Markov Cluster    
	    mcl = run_mcl(cosine_sim ,inflation = 1.5)
	    mcl_cluster = get_clusters(mcl)
	    mcl_labels, mcl_filter_labels , mcl_filter_cluster= label_mcl(mcl_cluster, len(cosine_sim))
	    
	    
	    centroid_list, centroid_word_list, score_list, centroid_name_list, size_list = find_centroid_score(tf, mcl_cluster, feature_names, topic_labels)
	    tweet_id_list, author_list = get_tweet_id(sample_doc, mcl_cluster)
	    write_event(date,'Markov Cluster', feature_names, centroid_list, centroid_word_list, score_list, centroid_name_list, size_list, tweet_id_list, author_list)
	    
	    # cluster_score
	    levels = ['topic', 'minor_class', 'major_class']
	    labels_true = [topic_labels, minor_class_labels, major_class_labels]
	    description = ['DBSCAN','Affinity Propagation','Markov Cluster']
	    labels = [db_labels,aff_labels,mcl_labels]
	    for i in range(3):
	        for j in range(3):
	            cluster_score(date, levels[i], description[j], labels_true[i], labels[j],topic_dense)
	bar.finish()
