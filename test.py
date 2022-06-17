import pygenalg
from psuedologger import psuedologger
import json
import numpy as np
import pandas as pd
from import_data import load_from_dir

log = psuedologger(file_out='log.out', file_lvl=10)

chr = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1]], dtype=float)
feats = np.array([[0,0,0,0], [5,5,5,5], [-10,-10,-10,-10]], dtype=float)
lbls = np.array([1,0,1], dtype=float)


feats = pd.read_csv('datasets/driving/indv627-all_tracks-all_track-training.csv')
feats.drop(columns=feats.columns[0], axis=1, inplace=True)
feats.drop(columns=feats.columns[0], axis=1, inplace=True)
lbls = feats.pop('Medicated')



test_feats = pd.read_csv('datasets/driving/indv627-all_tracks-all_track-test.csv')
test_feats.drop(columns=test_feats.columns[0], axis=1, inplace=True)
test_feats.drop(columns=test_feats.columns[0], axis=1, inplace=True)
test_lbls = test_feats.pop('Medicated')

cfg = {'evaluator':'logisticRegressionEvaluator',\
       'rep':'vector',
       'length':58,
       'min':-10.0,\
       'max':10.0,\
       'dtype':float,\
       'log':log,\
       'n_runs':50,\
       'n_gens':200,\
       'train_feats':feats,\
       'train_lbls':lbls, \
       'test_feats':test_feats,\
       'test_lbls':test_lbls,\
       'store_each_gen':True,\
       'standardize':True
      }

ga = pygenalg.geneticAlgorithm(config=cfg)

x = ga.run()
