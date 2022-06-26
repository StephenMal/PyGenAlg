import pygenalg
from psuedologger import psuedologger
import json, os
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm

incl_all_indvs = True
incl_spc_indvs = True

incl_all_track = False
incl_turns = True

incl_all_tracks = True
incl_spc_tracks = True

log = psuedologger(file_out='log.out', file_lvl=10)

train_file = sys.argv[1]
try:
    gene_size = sys.argv[2]
except:
    gene_size = 16


test_file = train_file.replace('training', 'test')

split = train_file.split('-')

dir_path = f'results/bga-{split[0]}-{split[1]}-{split[2]}/'

if split[0] == 'all_indvs':
    if not incl_all_indvs:
        print('Exiting')
        exit()
elif not incl_spc_indvs:
    print('Exiting')
    exit()

if split[1] == 'all_tracks':
    if not incl_all_tracks:
        print('Exiting')
        exit()
elif not incl_spc_tracks:
    print('Exiting')
    exit()

if split[2] == 'all_track':
    if not incl_all_track:
        print('Exiting')
        exit()
elif not incl_turns:
    print('Exiting')
    exit()

# Skip if we already got this directory results
dir_path = f'results_new/bga-{split[0]}-{split[1]}-{split[2]}/'

if os.path.isdir(dir_path):
    if 'config.json' in os.listdir(dir_path):
        print('Exiting')
        exit()
    if os.path.exists(os.path.join(dir_path, 'populations')) and \
                len(os.listdir(os.path.join(dir_path, 'populations'))) == 100:
        print('Exiting')
        exit()


try:
    # Get the training file
    train = os.path.join('datasets/driving/',train_file)

    # Get the test file
    test_file_n = '-'.join(split[:-1])+'-test.csv'
    test = os.path.join('datasets/driving/',test_file)

    # Read in the training data
    feats = pd.read_csv(train)
    feats.drop(columns=feats.columns[0], axis=1, inplace=True)
    feats.drop(columns=feats.columns[0], axis=1, inplace=True)
    lbls = feats.pop('Medicated')

    # Read in the testing data
    test_feats = pd.read_csv(test)
    test_feats.drop(columns=test_feats.columns[0], axis=1, inplace=True)
    test_feats.drop(columns=test_feats.columns[0], axis=1, inplace=True)
    test_lbls = test_feats.pop('Medicated')

    # Parameters
    cfg = {'evaluator':'logisticRegressionEvaluator',\
           'rep':'binary',
           'n_genes':59,\
           'gene_size':gene_size,\
           'length':59,\
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
           'standardize':True,\
           'has_constant':True,\
           'dir_path':dir_path,\
           'disable_tqdm':True
          }

    ga = pygenalg.geneticAlgorithm(config=cfg)
    x = ga.run()
except KeyboardInterrupt:
    print('Exiting')
    exit()
except Exception as e:
    try:
        tqdm.write(f'Failed {split[0]}, {split[1]}, {split[2]}')
    except:
        tqdm.write('failed')
    tqdm.write(str(e))
    exit()
