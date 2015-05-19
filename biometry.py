#!/usr/bin/env python
import bz2
import csv
import numpy as np
import sklearn.metrics as mtr
import sys
import time
import matplotlib.pyplot as plt

import hashtable

N_COHORTS = 500

def load_model_key(filename, model_ids):
    """Loads a model key to match order of model_ids

    Parameters
    ----------
    filename : string
        Path to target_speaker_models.csv
    model_ids : list
        List of model ivectorids from model_ivectors.csv

    Returns
    -------
    y : array, shaped('n_model_ivecs',)
        Array with each entry the target_speaker_modelid, in
        same order as model_ids
    """
    # load a reverse lookup from ivectorid->target speaker model
    id_speaker_lookup = {}
    for row in csv.DictReader(open(filename, 'rb')):
        for e in row['model_ivectorids[5]'].split():
            id_speaker_lookup[e] = row['target_speaker_modelid']

    # convert to a simple key array for model ivectors
    y = []
    for model_id in model_ids:
        y.append( id_speaker_lookup[ model_id ] )

    return np.array(y)

def load_ivectors(filename):
    """
    Loads ivectors

    Parameters
    ----------
    filename : string
        Path to ivector files (e.g. dev_ivectors.csv)

    Returns
    -------
    ids : list
        List of ivectorids
    durations : array, shaped('n_ivectors')
        Array of durations for each ivectorid
    ivectors : array, shaped('n_ivectors', 600)
        Array of ivectors for each ivectorid
    """
    ids = []
    durations = []
    ivectors = []
    
    for row in csv.DictReader(open(filename, 'rb')):
        ids.append(row['ivectorid'])
        durations.append(float(row['duration_secs']))
        ivectors.append(np.fromstring(row['values[600]'], count=600, sep=' ',
                                      dtype=np.float32))

    return ids, np.array(durations, dtype=np.float32), np.vstack(ivectors)

def z_norm(scores, avg_model_ivec, avg_model_names, model_ivec, model_key,
           n_norm=-1, use_lsh=True):
    
    if n_norm > 0 and use_lsh:
        knn = hashtable.LSHkNN.create_voronoi_lsh(int(np.ceil(n_norm / 50.)), 150, model_ivec)
    
    model_key_aux = np.array(model_key)
    
    for i_model, (key, model) in enumerate(zip(avg_model_names,
                                               avg_model_ivec)):
        if n_norm > 0:
            if use_lsh:
                z_distances, similar_trials = knn.knn_query(n_norm, model)
                is_not_self = (model_key_aux[similar_trials] != key)
                z_scores = 1 - z_distances[is_not_self]
            else:
                is_not_self = (model_key_aux != key)
                z_scores = np.dot(model_ivec[is_not_self],
                                  model[:, np.newaxis])[:, 0]
                z_scores = - np.sort(- z_scores)[:n_norm]
        else:
            is_not_self = (model_key_aux != key)
            z_scores = np.dot(model_ivec[is_not_self],
                              model[:, np.newaxis])[:, 0]
        
        z_mean = np.mean(z_scores)
        z_std = np.std(z_scores)
        
        if z_std == 0:
            z_std = 1
        
        scores[:, i_model] -= z_mean
        scores[:, i_model] /= z_std

def t_norm(scores, avg_model_ivec, test_ivec, use_t_norm_cohorts=False,
           use_t_norm_rank=False, t_norm_record_similar=False, n_norm=-1,
           use_lsh=True):
    
    if t_norm_record_similar:
        similar = np.empty((test_ivec.shape[0], 5), dtype=np.int)
    
    if n_norm > 0:
        # Use LSH to perform t-normalization
        if use_t_norm_cohorts:
            if use_lsh:
                knn = hashtable.LSHkNN.create_voronoi_lsh(10, 20,
                                                          avg_model_ivec)
            
            cohorts = np.empty((avg_model_ivec.shape[0], N_COHORTS),
                               dtype=np.int)
            
            for i_model, model in enumerate(avg_model_ivec):
                if use_lsh:
                    _, similar_models = zip(*knn.knn_query(N_COHORTS + 1,
                                                           model))
                else:
                    similar_models = np.argsort(np.dot(avg_model_ivec,
                                                       model[:, np.newaxis])
                                                [:, 0])[:N_COHORTS + 1]
                
                cohorts[i_model] = \
                    filter(lambda i: i != i_model, similar_models)[:N_COHORTS]
        elif use_lsh:
            knn = hashtable.LSHkNN.create_voronoi_lsh(int(np.ceil(n_norm / 50.)), 25, avg_model_ivec)
    
    for i_data, data in enumerate(test_ivec):
        if n_norm > 0 and use_t_norm_cohorts:
            t_scores = np.dot(test_ivec[np.newaxis, i_data],
                              avg_model_ivec[cohorts].transpose())
            
            if use_t_norm_rank:
                scores[i_data] = (scores[i_data][:, np.newaxis] >=
                                  t_scores).mean(axis=1)
            else:
                t_mean = np.mean(t_scores, axis=1)
                t_std = np.std(t_scores, axis=1)
                scores[i_data] -= t_mean
                scores[i_data] /= t_std
        else:
            if n_norm > 0:
                if use_lsh:
                    t_distances, similar_models = knn.knn_query(n_norm, data)
                    t_scores = 1 - t_distances
                else:
                    t_scores = np.dot(avg_model_ivec, data[:, np.newaxis])[:, 0]
                    similar_models = np.argsort(- t_scores)[:n_norm]
                    t_scores = t_scores[similar_models]
            else:
                similar_models = range(avg_model_ivec.shape[0])
                t_scores = np.dot(avg_model_ivec, data[:, np.newaxis])[:, 0]
            
            if t_norm_record_similar:
                similar[i_data] = similar_models[:5]
            
            if use_t_norm_rank:
                scores[i_data] = (scores[i_data][:, np.newaxis] >=
                                  t_scores[np.newaxis, :]).mean(axis=1)
            else:
                t_mean = np.mean(t_scores)
                t_std = np.std(t_scores)
                scores[i_data] -= t_mean
                scores[i_data] /= t_std
    
    if t_norm_record_similar:
        return similar

def generate_models(model_ivec, model_key, m, W):
    # create a speaker model by taking mean over i-vectors
    # from that speaker (in this case each should have 5)
    model_keys = np.unique(model_key)
    avg_model_ivec = np.zeros((len(model_keys), model_ivec.shape[1]))
    avg_model_names = []
    
    for i, key in enumerate(model_keys):
        avg_model_ivec[i] = np.mean(model_ivec[model_key == key], axis=0)
        avg_model_names.append(key)
    
    # project the avg model i-vectors into unit sphere
    avg_model_ivec /= np.sqrt(np.sum(avg_model_ivec ** 2, axis=1)) \
        [:, np.newaxis]
    
    return avg_model_ivec, avg_model_names

def load_trial_keys(filename):
    with open(filename, "r") as arq:
        a = [i.strip().split("\t") for i in arq.readlines()[1:]]
    
    trial_model_id, trial_segment_id, trial_target, trial_set = zip(*a)
    trial_model_id = np.array(trial_model_id)
    trial_segment_id = np.array(trial_segment_id)
    trial_target = np.array([i == "target" for i in trial_target])
    trial_set = np.array(trial_set)
    #trial_is_eval = (trial_set == "eval")
    
    return trial_target, trial_set

def eer(scores, labels, trial_set):
    trial_set_keys = ["eval"]#np.unique(trial_set)
    
    for key in trial_set_keys:
        aux = (trial_set == key)
        pfa, tpr, thresholds = mtr.roc_curve(labels[aux], scores[aux])
        pmiss = 1 - tpr
        i = np.argmin(np.abs(pfa - pmiss))
        
        yield key, (pfa[i] + pmiss[i]) / 2., pfa[i], pmiss[i], thresholds[i]

def min_dcf(scores, labels, trial_set):
    trial_set_keys = ["eval"]#np.unique(trial_set)
    
    for key in trial_set_keys:
        #p_misses(thresh=t) + (100 * fpr(thresh=t))
        aux = (trial_set == key)
        fpr, tpr, _ = mtr.roc_curve(labels[aux], scores[aux])
        pmiss = 1 - tpr
        
        yield key, min(map(lambda a: 100 * a[0] + a[1], zip(fpr, pmiss)))

def det_graph(scores, labels, trial_set, marker):
    #p_misses(thresh=t) + (100 * fpr(thresh=t))
    fpr, tpr, _ = mtr.roc_curve(labels[trial_set == "eval"], scores[trial_set == "eval"])
    pmiss = 1 - tpr
    
    plt.loglog(fpr, tpr, marker)

def main():
    # load ivector ids, durations and ivectors (as row vectors)
    dev_ids, dev_durations, dev_ivec = load_ivectors("data/dev_ivectors.csv")
    model_ids, model_durations, model_ivec = \
        load_ivectors("data/model_ivectors.csv")
    test_ids, test_durations, test_ivec = \
        load_ivectors("data/test_ivectors.csv")
    
    # load model key corresponding to the same ordering as model_ids
    model_key = load_model_key('data/target_speaker_models.csv', model_ids)
    
    # compute the mean and whitening transformation over dev set only
    m = np.mean(dev_ivec, axis=0)
    S = np.cov(dev_ivec, rowvar=0)
    D, V = np.linalg.eig(S)
    W = (1 / np.sqrt(D) * V).transpose().astype('float32')
    
    # center and whiten all i-vectors
    dev_ivec = np.dot(dev_ivec - m, W.transpose())
    test_ivec = np.dot(test_ivec - m, W.transpose())
    model_ivec = np.dot(model_ivec - m, W.transpose())
    
    # project all i-vectors into unit sphere
    model_ivec /= np.sqrt(np.sum(model_ivec ** 2, axis=1))[:, np.newaxis]
    dev_ivec /= np.sqrt(np.sum(dev_ivec ** 2, axis=1))[:, np.newaxis]
    test_ivec /= np.sqrt(np.sum(test_ivec ** 2, axis=1))[:, np.newaxis]
    
    avg_model_ivec, avg_model_names = \
        generate_models(model_ivec, model_key, m, W)
    
    # compute inner product of all avg model ivectors and test ivectors
    # since everything is already unit length this is the cosine distance
    #scores = np.zeros((avg_model_ivec.shape[0], test_ivec.shape[0]))
    #for i in xrange(avg_model_ivec.shape[0]):
    #    for j in xrange(test_ivec.shape[0]):
    #        scores[i, j] = np.inner(avg_model_ivec[i], test_ivec[j])

    # The above can be computed quickly doing matrix multiplication but
    # note that each trial is independent and scores cannot computed based
    # on other avg model ivectors nor scores adjusted based on other scores
    
    #scores = np.dot(avg_model_ivec, test_ivec.transpose())
    scores = np.dot(test_ivec, avg_model_ivec.transpose())
    #scores = np.divide(scores - scores.mean(axis=1)[:, np.newaxis], scores.std(axis=1)[:, np.newaxis])
    #z_aux = np.dot(avg_model_ivec, avg_model_ivec.transpose())
    
    trial_target, trial_set = load_trial_keys("data/ivec14_sre_trial_key_release.tsv")
    avg_model_names_indexes = np.argsort(np.array(avg_model_names))
    test_ids_indexes = np.argsort(np.array(test_ids))
    
    print "No normalization"
    
    score_list = scores[test_ids_indexes].T[avg_model_names_indexes].reshape(-1)
    #det_graph(score_list, trial_target, trial_set, "r.")
    print "EER", list(eer(score_list, trial_target, trial_set))
    print "minDCF", list(min_dcf(score_list, trial_target, trial_set))
    print
    
    use_t_norm = True
    use_z_norm = True
    use_lsh = True
    
    for n_norm in [10, 25, 50, 100, 250, 500, 1000, -1]:
        for use_lsh in [True, False]:
            n_scores = scores.copy()
            
            begin = time.time()
            
            if use_z_norm:
                z_norm(n_scores, avg_model_ivec, avg_model_names, model_ivec,
                       model_key, n_norm=n_norm, use_lsh=use_lsh)
            
            if use_t_norm:
                t_norm(n_scores, avg_model_ivec, test_ivec, n_norm=n_norm,
                       use_lsh=use_lsh)
            
            end = time.time()
            
            print "ZT-norm (%d, %s)" % (n_norm, use_lsh)
            print "Time", (end - begin)
            
            n_score_list = n_scores[test_ids_indexes]. \
                T[avg_model_names_indexes].reshape(-1)
            print "EER", list(eer(n_score_list, trial_target, trial_set))
            print "minDCF", list(min_dcf(n_score_list, trial_target, trial_set))
            print
            
            sys.stdout.flush()
            
            det_graph(score_list, trial_target, trial_set, "r-")
            det_graph(n_score_list, trial_target, trial_set, "b-")
            plt.xlim(1e-3, 1)
            plt.ylim(1e-1, 1)
            plt.savefig("zt_norm_%d_%s.png" % (n_norm, use_lsh))
            plt.cla()

    
    # format and order scores for submission
    #    the format is a single score as ASCII formated float
    #    per line ordered lexigraphically by model_name then test_id
    #
    # we have this data
    #    avg_model_names (order of rows)
    #    test_ids (order of columns)
    #    scores (should be 1306 x 9634 for a total of 12,582,004 scores)
    #
    # so sort and output to a bz2 file
    #aux1 = np.empty(avg_model_names_indexes.shape, dtype=np.int)
    #aux1[avg_model_names_indexes] = np.arange(avg_model_names_indexes.size)
    """
    #f = bz2.BZ2File('myscores.txt.bz2', 'w')
    f = open('myscores_norm.txt', 'w')
    #print aux1[aux[test_ids_indexes[:20]]]
    for i in avg_model_names_indexes:
        for j in test_ids_indexes:
            f.write('%f\n' % scores[j, i])
    
    f.close()
    
    print "ZT-norm"
    
    score_list = scores[test_ids_indexes].T[avg_model_names_indexes].reshape(-1)
    det_graph(score_list, trial_target, trial_set, "b.")
    print "EER", list(eer(score_list, trial_target, trial_set))
    print "minDCF", list(min_dcf(score_list, trial_target, trial_set))
    
    #plt.xlim(1e-5, .1)
    #plt.ylim(1e-5, .1)
    plt.show()
    """

if __name__=='__main__':
    main()
