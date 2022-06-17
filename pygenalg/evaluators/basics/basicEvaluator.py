from ...basics import basicComponent
from ...indvs.basics.basicIndividual import basicIndividual
from ...exceptions import *
import sys


class basicEvaluator(basicComponent):

    __slots__ = {'maximize':'Whether or not the objective is to maximize',\
                 'dynamic':'Whether or not the objective changes',\
                 'gmax_indv':'The individual in current gen with the highest fitness',\
                 'gmax_fit':'The highest fitness found in current gen',\
                 'gmin_indv':'The individual in current gen with the lowest fitness',\
                 'gmin_fit':'The lowest fitness found in current gen',\
                 'rmax_indv':'The individual in current run with the highest fitness',\
                 'rmax_fit':'The highest fitness found in current run',\
                 'rmin_indv':'The individual in current run with the lowest fitness',\
                 'rmin_fit':'The lowest fitness found in current run',\
                 'smax_indv':'The individual from all runs with the highest fitness',\
                 'smax_fit':'The highest fitness found from all runs',\
                 'smin_indv':'The individual from all runs with the lowest fitness',\
                 'smin_fit':'The lowest fitness found from all runs'}

    DEFAULT_maximize = None
    DEFAULT_dynamic = False

    def __init__(self, *args, **kargs):

        self.maximize, self.dynamic = None, None
        # Generation Max
        self.gmax_indv, self.gmin_indv = None, None
        self.gmax_fit, self.gmin_fit = float('-inf'), float('inf')
        # Run max
        self.rmax_indv, self.rmin_indv = None, None
        self.rmax_fit, self.rmin_fit = float('-inf'), float('inf')
        # Super max (overall)
        self.smin_indv, self.smax_indv = None, None
        self.smax_fit, self.smin_fit = float('-inf'), float('inf')

        super().__init__(*args, **kargs)

        if self.__class__ == basicEvaluator:
            self.set_params(**kargs)

    def set_maximize(self, maximize):
        if not isinstance(maximize, bool):
            raise TypeError('maximize should be a boolean')
        self.maximize = maximize

    def set_dynamic(self, dynamic):
        if not isinstance(dynamic, bool):
            raise TypeError('maximize should be a boolean')
        self.dynamic = dynamic

    def wipe_minmax(self, level=0):

        # Generation Max
        if level == 0 or level == 1:
            self.gmax_indv, self.gmin_indv = None, None
            self.gmax_fit, self.gmin_fit = float('-inf'), float('inf')

        # Run max
        if level == 0 or level == 2:
            self.rmax_indv, self.rmin_indv = None, None
            self.rmax_fit, self.rmin_fit = float('-inf'), float('inf')

        # Super max (overall)
        if level == 0 or level == 3:
            self.smin_indv, self.smax_indv = None, None
            self.smin_fit, self.smax_fit = float('-inf'), float('inf')

    def get_maxfit(self, level=0):
        if level == 0:
            return (self.gmax_fit, self.rmax_fit, self.smax_fit)
        elif level == 1:
            return self.gmax_fit
        elif level == 2:
            return self.rmax_fit
        elif level == 3:
            return self.smax_fit

    def get_minfit(self, level=0):
        if level == 0:
            return (self.gmin_fit, self.rmin_fit, self.smin_fit)
        elif level == 1:
            return self.gmin_fit
        elif level == 2:
            return self.rmin_fit
        elif level == 3:
            return self.smin_fit


    def set_max_indv(self, indv, fit=None, level=0):
        ''' Replaces the max individual recorded in evaluator '''
        if not isinstance(indv, basicIndividual):
            if isinstance(indv, dict):
                try:
                    indv = self.unpack_component(indv)
                except:
                    raise ValueError('Passed dict, but could not unpack')
            else:
                raise TypeError('max_indv should be derivative of base individual')

        # Get fit if not provided
        if fit is None:
            fit = indv.get_fit()

        # Sees if best in current gen
        if level == 0 or level == 1:
            if fit > self.gmax_fit:
                self.gmax_indv = indv.pack()
                self.gmax_fit = fit

        # Sees if best in current run
        if level == 0 or level == 2:
            if fit > self.rmax_fit:
                self.rmax_indv = indv.pack()
                self.rmax_fit = fit

        # See if better than best seen of all runs
        if level == 0 or level == 3:
            if fit > self.smax_fit:
                self.smax_indv = indv.pack()
                self.smax_fit = fit


    def set_min_indv(self, indv, fit=None, level=0):
        ''' Replaces the min individual recorded in evaluator '''
        if not isinstance(indv, basicIndividual):
            if isinstance(indv, dict):
                try:
                    indv = self.unpack_component(indv)
                except:
                    raise ValueError('Passed dict, but could not unpack')
            else:
                raise TypeError('min_indv should be derivative of base individual')

        # Get fit if not provided
        if fit is None:
            fit = indv.get_fit()

        # Sees if best in current gen
        if level == 0 or level == 1:
            if fit < self.gmin_fit:
                self.gmin_indv = indv.pack()
                self.gmin_fit = fit
        # Sees if best in current run
        if level == 0 or level == 2:
            if fit < self.rmin_fit:
                self.rmin_indv = indv.pack()
                self.rmin_fit = fit
        # See if better than best seen of all runs
        if level == 0 or level == 3:
            if fit < self.smin_fit:
                self.smin_indv = indv.pack()
                self.smin_fit = fit


    def set_params(self, **kargs):

        if 'maximize' in kargs:
            self.set_maximize(kargs.get('maximize'))
        elif self.maximize is None and ('maximize' in self.config or \
                                        self.DEFAULT_maximize is not None):
            self.set_maximize(\
                self.config.get('maximize', self.DEFAULT_maximize, dtype=bool))

        if 'dynamic' in kargs:
            self.set_dynamic(kargs.get('dynamic'))
        elif self.dynamic is None and ('dynamic' in self.config or \
                                    self.DEFAULT_dynamic is not None):
            self.set_dynamic(self.config.get('dynamic', \
                                            self.DEFAULT_dynamic, dtype=bool))

        for prefix in ('g', 's', 'r'):
            if f'{prefix}max_indv' in kargs:
                self.set_max_indv(kargs.get('max_indv'))
            if f'{prefix}min_indv' in kargs:
                self.set_min_indv(kargs.get('min_indv'))

        super().set_params(**kargs)

    def evaluate(self, indv, **kargs):
        raise NotImplementedError('basicEvaluator does not implement evaluate')

    def evaluate_batch(self, btch, **kargs):

        # Track minimum and maximum individuals
        update_minmax = kargs.get('update_minmax', True)
        if update_minmax:
            minfit, maxfit = (None, float('-inf')), (None, float('inf'))

        # Create new key args with udpate_minmax set to false (to avoid repeated)
        kargs = kargs.copy()
        kargs['update_minmax'] = False

        # Run through individuals
        for indv in btch:
            # Evaluate
            self.evaluate(indv, **kargs)

            # Track min/max
            if update_minmax:
                if indv.get_fit() < minfit[1]:
                    minfit = (indv, indv.get_fit())
                elif indv.get_fit() > minfit[1]:
                    maxfit = (indv, invd.get_fit())

        # replace
        if update_minmax:
            self.set_max_indv(maxfit[0], fit=maxfit[1])
            self.set_min_indv(minfit[0], fit=minfit[1])

    def get_max(self, level=2):
        if level == 0:
            return (self.gmax_indv, self.rmax_indv, self.smax_indv)
        elif level == 1:
            return self.gmax_indv
        elif level == 2:
            return self.rmax_indv
        elif level == 3:
            return self.smax_indv

    def get_min(self, level=2):
        ''' Returns the minimum individual '''
        if level == 0:
            return (self.gmin_indv, self.rmin_indv, self.smin_indv)
        elif level == 1:
            return self.gmin_indv
        elif level == 2:
            return self.rmin_indv
        elif level == 3:
            return self.smin_indv

    def get_best(self, level=2):
        ''' Returns the worst individual (based of maximize)'''

        if self.maximize is True:
            return self.get_max(level=level)
        elif self.maximize is False:
            return self.get_min(level=level)
        else:
            raise ValueError('maximize should be a bool')
        raise Exception

    def get_worst(self, level=2):
        ''' Returns the best individual (based of maximize)'''
        if self.maximize is True:
            return self.get_min(level=level)
        elif self.maximize is False:
            return self.get_max(level=level)
        else:
            raise ValueError('maximize should be a bool')

    # Creates a competitive template using a greedy algorithm inspired by the
    #   Messy Genetic Algorithm [1].  Creates a chromosome, runs through it
    #   in random ordering per sweep, and applies mutations.  Keeps if it is
    #   was a positive change
    def make_template(self, **kargs):
        self.log.exception('make_template Not Implemented yet', \
                                err=NotImplementedError)

    
    def analyze_indv(self, indv):
        ''' Returns a dictionary containing information of an indv solution '''
        raise NotImplementedError

    def pack(self, **kargs):
        if self.maximize is None:
            raise MissingPackingValue(argname='maximize')
        dct = super().pack(**kargs)

        if kargs.get('incl_defs', False):
            dct.update({'maximize':self.bool_to_int(self.maximize),\
                        'dynamic':self.bool_to_int(self.dynamic)})
        else:
            if self.maximize != self.DEFAULT_maximize:
                dct['maximize'] = self.maximize
            if self.dynamic != self.DEFAULT_dynamic:
                dct['dynamic'] = self.dyanmic

        indvs = {}
        if self.max_indv is not None:
            indvs['max_indv'] = \
                    self.max_indv.pack(**kargs)
        if self.min_indv is not None:
            indvs['min_indv'] = \
                    self.min_indv.pack(**kargs)
        if self.smax_indv is not None:
            indvs['smax_indv'] = \
                    self.smax_indv.pack(**kargs)
        if self.smin_indv is not None:
            indvs['smin_indv'] = \
                    self.smin_indv.pack(**kargs)

    @classmethod
    def unpack(cls, dct):
        if 'maximize' not in dct and self.DEFAULT_maximize is None:
            raise MissingPackingValue(argname='maximize')
        if 'dynamic' not in dct and self.DEFAULT_dynamic is None:
            raise MissingPackingValue(argname='dynamic')
        return super().unpack(dct)

'''
[1]  Goldberg, David E., Bradley Korb, and Kalyanmoy Deb. "Messy genetic
algorithms: Motivation, analysis, and first results." Complex systems
3.5 (1989): 493-530.
'''
