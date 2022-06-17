import json
import pygenalg
from pygenalg.populations.basics.basicPopulation import basicPopulation
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import os
import math
from statistics import mean, stdev

results_dir = 'results2'

dirs = os.listdir(results_dir)

sols = []
for dir in tqdm(dirs, desc='Getting solution info', unit='run-set', colour='white'):

    # Skip if not a directory
    if not os.path.isdir(os.path.join(results_dir, dir)):
        continue
        
    try:
        split = dir.split('_')
        rep, indv, track, turn = split[0], split[1], split[2], split[3]
    except:
        continue

    try:
        with open(os.path.join(results_dir, dir, 'solution_indv.json'),'r') as F:
            dct = json.load(F)
            dct['rep'] = rep
            dct['indv'] = indv
            dct['track'] = track
            dct['turn'] = turn
            keys = 'Velocity,VelocityMagnitude,LateralVelocity,VerticalVelocity,Steer,SteerChange,SteerMagnitude,SteerAbsChange,Brake,BrakeMagnitude,BrakeChange,BrakeAbsChange,LaneOffset,LaneOffsetMagnitude,LaneOffsetChange,LaneOffsetAbsChange,LatAcc,LonAcc,Roll,RollMagnitude,RollChange,RollAbsChange,Pitch,PitchMagnitude,PitchChange,PitchAbsChange,EngineRPM,EngineTorque,Throttle'.split(',')
            w = dct['attrs'].pop('weights')
            for key, weight in zip(keys, w):
                dct[key] = weight
            sols.append(dct)
    except:
        continue
pd.json_normalize(sols).to_csv(os.path.join(results_dir, 'final_solutions.csv'))

# Iterate through the directories
for dir in tqdm(dirs, desc='Getting run info', unit='run-set', colour='white', leave=False):

    # Skip if not a directory
    if not os.path.isdir(os.path.join(results_dir, dir)):
        continue

    # Get the populations results
    pop_dcts = os.path.join(results_dir, dir, 'populations')
    # Split into run files and solution files

    runs = [file_n for file_n in os.listdir(pop_dcts) if 'run' in file_n]
    sols = [file_n for file_n in os.listdir(pop_dcts) if 'sol' in file_n]

    ''' Handles runs '''

    # Get the runfile
    run_out = open(os.path.join(results_dir, dir, 'runstats.csv'), 'w')
    run_out.write('rep,indv,track,turn,run,gen,avg_fit,fit_95CI,fit_stdev,fit_stdev_n,best_train_acc,best_test_acc,avg_train_acc,avg_train_acc_std,avg_test_acc,avg_test_acc_std\n')
    # Iterate through the run files
    for nrun, runf in enumerate(tqdm(runs, desc='Going through run files',unit='run-file', colour='blue', leave=False)):
        # Read in the lines of the runfile
        with open(os.path.join(pop_dcts, runf),'r') as F:
            lines = F.readlines()

        if len(lines) < 49:
            continue

        # Iterate through the runs
        for ngen, runjson in enumerate(tqdm(lines, desc='Going through generations', unit='run', colour='green', leave=False)):
            pop = basicPopulation.unpack_component(json.loads(runjson))
            avg, CI = pop.mean(z=1.96)
            attrs = pop.consolidate_attrs()
            # Training acc
            best_train_acc = max(attrs['train_acc'])
            avg_train_acc = mean(attrs['train_acc'])
            std_train_acc = stdev(attrs['train_acc'])
            # Testing acc
            best_test_acc = max(attrs['test_acc'])
            avg_test_acc = mean(attrs['test_acc'])
            std_test_acc = stdev(attrs['test_acc'])
            # Output the data
            run_out.write(f'{rep},{indv},{track},{turn},'+\
                          f'{nrun},{ngen},{avg},{CI},{pop.stdev()},'+\
                          f'{pop.nstdev()},{best_train_acc},{best_test_acc},'+\
                          f'{avg_train_acc},{std_train_acc},{avg_test_acc},'+\
                          f'{std_test_acc}\n')
    run_out.close()

    ''' Handles solutions '''

    soljsons = []
    for solf in tqdm(sols, desc='Loading sol-files', unit='sol-file', colour='green', leave=False):
        try:
            with open(os.path.join(pop_dcts, solf),'r') as F:
                soljsons.append(json.load(F))
        except:
            continue
    pd.json_normalize(soljsons).to_csv(os.path.join(results_dir, dir, 'solstats.csv'))
    wfile = open(os.path.join(results_dir, dir, 'best_weights.csv'), 'w')
    wfile.write('Rep,indv,track,turn,Velocity,VelocityMagnitude,LateralVelocity,VerticalVelocity,Steer,SteerChange,SteerMagnitude,SteerAbsChange,Brake,BrakeMagnitude,BrakeChange,BrakeAbsChange,LaneOffset,LaneOffsetMagnitude,LaneOffsetChange,LaneOffsetAbsChange,LatAcc,LonAcc,Roll,RollMagnitude,RollChange,RollAbsChange,Pitch,PitchMagnitude,PitchChange,PitchAbsChange,EngineRPM,EngineTorque,Throttle\n')
    for solj in tqdm(sols, desc='Gathering weights', unit='sol-json',colour='green', leave=False):
        with open(os.path.join(pop_dcts, solj), 'r') as F:
            dct = json.load(F)
        wfile.write(f'{rep},{indv},{track},{turn},'+','.join([str(w) for w in dct['attrs']['weights']])+'\n')


'''
Needed:
 - Average per generation
 - Stdev per generation
 - Train Acc per generation
 - Test Acc per generation
 - Fitness per gen
'''
