#!/usr/bin/env python
import bz2
import csv
import numpy as np

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
    model_ivec = np.dot(model_ivec - m, W.transpose())
    test_ivec = np.dot(test_ivec - m, W.transpose())
    
    # project all i-vectors into unit sphere
    dev_ivec /= np.sqrt(np.sum(dev_ivec ** 2, axis=1))[:, np.newaxis]
    model_ivec /= np.sqrt(np.sum(model_ivec ** 2, axis=1))[:, np.newaxis]
    test_ivec /= np.sqrt(np.sum(test_ivec ** 2, axis=1))[:, np.newaxis]
    
    # create a speaker model by taking mean over i-vectors
    # from that speaker (in this case each should have 5)
    avg_model_ivec = np.zeros((len(np.unique(model_key)), model_ivec.shape[1]))
    avg_model_names = []
    
    for i, key in enumerate(np.unique(model_key)):
        avg_model_ivec[i] = np.mean(model_ivec[model_key == key], axis=0)
        avg_model_names.append(key)
    
    # project the avg model i-vectors into unit sphere
    avg_model_ivec /= np.sqrt(np.sum(avg_model_ivec ** 2, axis=1)) \
        [:, np.newaxis]
    
    # compute inner product of all avg model ivectors and test ivectors
    # since everything is already unit length this is the cosine distance
    #scores = np.zeros((avg_model_ivec.shape[0], test_ivec.shape[0]))
    #for i in xrange(avg_model_ivec.shape[0]):
    #    for j in xrange(test_ivec.shape[0]):
    #        scores[i, j] = np.inner(avg_model_ivec[i], test_ivec[j])

    # The above can be computed quickly doing matrix multiplication but
    # note that each trial is independent and scores cannot computed based
    # on other avg model ivectors nor scores adjusted based on other scores
    scores = np.dot(avg_model_ivec, test_ivec.transpose())


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
    avg_model_names_indexes = np.argsort(np.array(avg_model_names))
    test_ids_indexes = np.argsort(np.array(test_ids))
    
    f = bz2.BZ2File('myscores.txt.bz2', 'w')
    
    for i in avg_model_names_indexes:
        for j in test_ids_indexes:
            f.write('%f\n' % scores[i,j])
    
    f.close()

if __name__=='__main__':
    main()
