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

#train_file, test_file = sys.argv[1], sys.argv[2]

#dir_path = f'results/bga_{split[0]}_{split[1]}_{split[2]}/'
files = []
for file_n in tqdm(os.listdir('datasets/driving'), leave=False):

    if 'training' not in file_n:
        continue

    split = file_n.split('-')

    if not incl_all_indvs:
        if plit[0] == 'all_indvs':
            continue
    elif not incl_spc_indvs:
        continue

    if split[1] == 'all_tracks':
        if not incl_all_tracks:
            continue
    elif not incl_spc_tracks:
        continue

    if not incl_all_track:
        if split[2] == 'all_track':
            continue
    elif not incl_turns:
        continue

    files.append(file_n)

for file_n in tqdm(files):
    for gene_size in (6, 12, 18):
        try:
            # Get the directory
            split = file_n.split('-')

            dir_path = f'results/bga-{split[0]}-{split[1]}-{split[2]}-{gene_size}/'

            # Skip if we already got this directory results
            if os.path.isdir(dir_path):
                if 'config.json' in os.listdir(dir_path):
                    print('H')
                    continue
                if os.path.exists(os.path.join(dir_path, 'populations')) and \
                            len(os.listdir(os.path.join(dir_path, 'populations'))) == 100:
                    print('I')
                    continue
            files.append(file_n)


            # Get the training file
            train = os.path.join('datasets/driving/',file_n)

            # Get the test file
            test_file_n = '-'.join(split[:-1])+'-test.csv'
            test = os.path.join('datasets/driving/',test_file_n)

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
                   'disable_tqdm':False
                  }

            ga = pygenalg.geneticAlgorithm(config=cfg)

            x = ga.run()
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            tqdm.write(f'Failed {dir_path}')
            continue
