import pygenalg
from psuedologger import psuedologger
import json
import numpy as np
import pandas as pd
from import_data import load_from_dir

log = psuedologger(file_out='log.out', file_lvl=10)
'''
cfg = {'evaluator':'numberMatchingEvaluator',\
       'rep':'proportional',
       'match_arr':[1,2,3,4,5,6,7,8,9],\
       'n_genes':10,\
       'length':100,
       'min':0,\
       'max':10,\
       'dtype':int,\
       'log':log,\
       'n_runs':50,\
       'n_gens':200,\
       'store_each_gen':True}
'''
cfg = {'evaluator':'numberMatchingEvaluator',\
       'rep':'floating',
       'match_arr':[0,1,2,3,4,5,6,7,8],\
       'n_genes':9,\
       'gene_size':4,\
       'length':250,
       'dtype':int,\
       'log':log,\
       'n_runs':50,\
       'n_gens':200,\
       'store_each_gen':True}

ga = pygenalg.geneticAlgorithm(config=cfg)

x = ga.run()
