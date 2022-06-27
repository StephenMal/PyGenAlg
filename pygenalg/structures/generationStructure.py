# PyGenAlg object imports
from .basics import basicStructure
# Get fixed populaiton (only valid type for this)
from ..populations.fixedPopulation import fixedPopulation

# Import basics
from ..populations.basics.basicPopulation import basicPopulation
from ..evaluators.basics.basicEvaluator import basicEvaluator
from ..genetic_operators.basics.basicOperator import basicOperator
from ..selectors.basics.basicSelector import basicSelector

# Dependency imports
from tqdm import tqdm, trange

# Built-in imports
from collections import namedtuple
from time import sleep
import sys, random, json, os

from statistics import mean, stdev

class generationStructure(basicStructure):

    __slots__ = ()

    gen_tpl = namedtuple('gen_results', \
                                ('gen', 'best_indv', 'worst_indv', 'avg_fit'))
    run_tpl = namedtuple('run_results', \
                                ('run', 'best_indv', 'worst_indv', 'gens'))
    tot_tpl = namedtuple('tot_results', \
                                ('runs', 'best_indv', 'worst_indv'))

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def run(self, **kargs):
        # Create output directory

        if 'dir_path' in kargs:
            outdir = self.create_dir(full_path=kargs.get('dir_path'))
            self.config.get('dir_path', outdir)
        elif 'dir_path' in self.config:
            outdir = self.create_dir(full_path=self.config.get('dir_path'))
        else:
            outdir = self.create_dir(prefix='results')
            self.config.get('dir_path', outdir)

        tqdm.write(f'Output Directory: {outdir}')

        self.log.set_file_out(os.path.join(outdir, 'log.out'), append=False)


        # Whether or not to disable the tqdm bar
        if 'disable_tqdm' in kargs:
            disable_tqdm = kargs.get('disable_tqdm')
            tqdm.write('A')
        else:
            disable_tqdm = self.config.get('disable_tqdm', False, dtype=bool)
            tqdm.write(f'B {disable_tqdm}')
        if not isinstance(disable_tqdm, bool):
            raise TypeError('disable_tqdm should be a bool')

        # Number of runs
        if 'n_runs' in kargs:
            n_runs = kargs.get('n_runs')
        else:
            n_runs = self.config.get('n_runs', 1, dtype=int, mineq=1)
        if not isinstance(n_runs, int):
            raise TypeError('n_runs should be an int')
        if not n_runs >= 1:
            raise ValueError('n_runs should be greater than 0')

        # Number of gens
        if 'n_gens' in kargs:
            n_gens = kargs.get('n_gens')
        else:
            n_gens = self.config.get('n_gens', 200, dtype=int, mineq=1)
        if not isinstance(n_gens, int):
            raise TypeError('n_gens should be an int')
        if not n_gens >= 1:
            raise ValueError('n_gens should be greater than 0')

        # Export population at each generation
        if 'store_each_gen' in self.config:
            store_each_gen = self.config.get('store_each_gen', dtype=bool)
        else:
            store_each_gen = kargs.get('store_each_gen', False)

        if 'store_gen_sum' in self.config:
            store_gen_sum = self.config.get('store_gen_sum', dtype=bool)
        else:
            store_gen_sum = kargs.get('store_gen_sum', False)

        if store_each_gen or store_gen_sum:
            gen_dir = self.create_dir(prefix=outdir, fol_name='populations')

        # Get n_runs & n_gens
        self.log.info(f'Starting runs ({n_runs} runs, {n_gens} gens)')

        # Get the evaluator
        evaluator = None
        if 'evaluator' in kargs:
            evaluator = self.get_evaluator(kargs.get('evaluator'))
        else:
            evaluator = self.get_evaluator(self.config.get('evaluator', \
                                                dtype=(str, basicEvaluator)))

        # Get the selector
        selector = None
        if 'selector' in kargs:
            selector = self.get_selector(kargs.get('selector'))
        else:
            selector = self.get_selector(self.config.get('selector', \
                                                         'tournamentSelector',\
                                                    dtype=(str, basicSelector)))

        # Get the genetic operators
        g_op = None
        if 'g_op' in kargs or 'genetic_operator' in kargs:
            g_op = self.get_genetic_operator(\
                            kargs.get('g_op', kargs.get('genetic_operator')))
        else:
            if 'g_op' in self.config:
                g_op = self.get_genetic_operator(\
                                    self.config.get('g_op', 'basicOperator',\
                                                    dtype=(str, basicOperator)))
            else:
                g_op = self.get_genetic_operator(self.config.get('genetic_operator',\
                                                            'basicOperator',\
                                                    dtype=(str, basicOperator)))

        # Stores results
        results = []

        # Create parents variable, however will be instantiated per generation
        parents, children = None, None

        # Generate the run loading bar
        runbar = trange(n_runs, unit = 'run', colour = '#7393B3', \
                        desc = 'Runs', \
                        disable = (disable_tqdm or n_runs < 2),\
                        leave=False)

        # Iterate through run
        for run in runbar:

            # Generate the gen loading bar
            self.log.info(f'Starting run #{run+1}/{n_runs}')

            # Create population
            self.log.debug('Creating population objects')
            parents = fixedPopulation(config=self.config,\
                                      log=self.log,\
                                      runsafe=self.runsafe)
            children = fixedPopulation(config=self.config,\
                                      log=self.log,\
                                      runsafe=self.runsafe)
            popsize = self.config.get('popsize', dtype=int, mineq=2)

            # Generate a starting population
            self.log.debug('0\t| Generating starting population')
            parents.generate()
            children.generate()

            # Evaluate initial parents
            self.log.debug('0\t| Evaluating initial population')
            evaluator.evaluate_batch(parents.get_poplst())

            # Select
            self.log.debug('0\t| Selecting parents from initial population')
            selected = selector.select_parents(parents,  n=popsize)

            # Create file storing this runs' pop
            if store_each_gen:
                popfile = open(os.path.join(gen_dir, f'run{run}.json'), 'w')
                json.dump(parents.pack(), popfile)
            if store_gen_sum:
                sumfile = open(os.path.join(gen_dir, f'gensum{run}.json'), 'w')



            # Iterate through the generations
            genbar = trange(1, n_gens+1, unit='gen', colour='#088F8F', \
                desc='Gens', disable = disable_tqdm, \
                leave=False)
            gen_lst = []
            for gen in genbar:

                self.log.debug(f'Starting gen #{gen}/{n_gens}')

                # Build new pop
                self.log.debug(f'{gen}\t| Creating children')
                g_op.create_children(selected, children)

                # Swap parents / children
                self.log.debug(f'{gen}\t| Swapping children & parents populations')
                parents, children = children, parents

                # Evaluate
                self.log.debug(f'{gen}\t| Evaluating new parents')
                evaluator.evaluate_batch(parents.get_poplst())

                best = evaluator.get_best(level=1)
                best_fit = best['fit']
                self.log.info(f'{gen}\t| Best fitness: ' + str(best_fit))
                genbar.set_postfix_str(f'Best Fit: {round(best_fit, 4)}')

                # Select
                self.log.debug(f'{gen}\t| Selecting children to create new generation')
                selected = selector.select_parents(parents,  n=popsize)

                # Store results regarding this generation
                gen_lst.append(\
                    self.gen_tpl(gen=gen, \
                                 best_indv=best,\
                                 worst_indv=evaluator.get_worst(level=1),\
                                 avg_fit=parents.mean()))

                # Wipe what is stored for this generation's best
                evaluator.wipe_minmax(level=1)

                # If storing population, do it
                if store_each_gen:
                    popfile.write('\n')
                    json.dump(parents.pack(), popfile)
                if store_gen_sum:
                    sumfile.write('\n')
                    attrs = parents.consolidate_attrs()
                    json.dump(\
                        {'mean':parents.mean(), 'ci':parents.mean(z=1.96)[1],\
                         'stdev':parents.stdev(), 'nstdev':parents.nstdev(),
                         'max':parents.max(), 'min':parents.min(),\
                         'mean_train_acc':mean(attrs['train_acc']),\
                         'stdev_test_acc':stdev(attrs['train_acc']),\
                         'mean_test_acc':mean(attrs['test_acc']),\
                         'stdev_test_acc':stdev(attrs['test_acc'])}, sumfile)

            best_fit = evaluator.get_best(level=3)['fit']
            runbar.set_postfix_str(f'Best Fit: {round(best_fit, 4)}')
            with open(os.path.join(outdir, 'solution_indv.json'), 'w') as F:
                json.dump(evaluator.get_best(level=3), F)

            if store_each_gen or store_gen_sum:
                with open(os.path.join(gen_dir, f'sol{run}.json'), 'w') as F:
                    json.dump(evaluator.get_best(level=3), F)

            genbar.reset()

            # Append all results from generation
            results.append(\
                self.run_tpl(run=run,\
                             best_indv=evaluator.get_best(level=2),\
                             worst_indv=evaluator.get_worst(level=2),\
                             gens=gen_lst))

            # Wipe best for run
            evaluator.wipe_minmax(level=2)

            if store_each_gen:
                popfile.close()
            if store_gen_sum:
                sumfile.close()

        genbar.close()
        runbar.close()

        # Output config file there (wait till end to include any defaults)
        with open(os.path.join(outdir, 'config.json'), 'w') as F:
            self.config.json_dump(F)
        with open(os.path.join(outdir, 'config_sum.txt'), 'w') as F:
            F.write(self.config.sumstr())

        res_tpl = self.tot_tpl(runs=results,\
                            best_indv=evaluator.get_best(level=3),\
                            worst_indv=evaluator.get_worst(level=3))

        del evaluator, selector, g_op, parents, children

        return res_tpl
