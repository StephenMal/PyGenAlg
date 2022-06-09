import pygenalg
from psuedologger import psuedologger
import json
import numpy as np
import pandas as pd

log = psuedologger(file_out='log.out', file_lvl=10)

chr = np.array([[1, 1, 1,1,1,1,1,1], [0, 1, 0,1,0,1,0,1]], dtype=float)
feats = np.array([[0,0,0,0], [5,5,5,5], [-10,-10,-10,-10]], dtype=float)
lbls = np.array([1,0,1], dtype=float)


cfg = {
    'evaluator':'logisticRegressionEvaluator',\
    'rep':'vector',
    'length':8,
    'min':-10.0,\
    'max':10.0,\
    'dtype':float,\
    'match_arr':[0,1,2,3,4,5,6,7,8,9],\
    'log':log,\
    'n_runs':1,\
    'n_gens':5,\
    'train_feats':feats,\
    'train_lbls':lbls
}

ga = pygenalg.geneticAlgorithm(config=cfg)

x = ga.run()

print(x)

print(json.dumps(x))
