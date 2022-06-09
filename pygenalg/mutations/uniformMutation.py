from .basics.basicMutation import basicMutation
from ..indvs.basics import basicChromosome, basicIndividual
import numpy as np
import random
import sys

try:
    import numba as nb
    from numba import prange
except:
    pass

class uniformMutation(basicMutation):

    __slots__ = ()

    use_nbjit_fxn = 'numba' in sys.modules
    DEFAULT_mutrate = 0.05

    def __init__(self, **kargs):

        super().__init__(**kargs)

        if self.__class__ == uniformMutation:
            self.set_params(**kargs)

    if 'numba' in sys.modules:
        @staticmethod
        @nb.jit(nopython=True, parallel=True)
        def _mutate(chromo, mutrate, minv, maxv, isint):
            if isint == True:
                for indx in prange(len(chromo)):
                    if np.random.rand() < mutrate:
                        chromo[indx] = np.random.randint(minv,high=maxv+1)
            else:
                for indx in prange(len(chromo)):
                    if np.random.rand() < mutrate:
                        chromo[indx] = np.random.uniform(minv, maxv)
            return chromo

    else:
        @staticmethod
        def _mutate(chromo, mutrate, minv, maxv, isint):
            mask = self.nprng.choice([0,1], \
                                     size=len(chromo),\
                                     p=(1-mutrate, mutrate))
            if isint == True:
                chromo[mask] = random.integers(minv, high=maxv+1)
            else:
                chromo[mask] = random.uniform(minv, high=maxv)
            return chromo

    def _mutate_chromo(self, items, **kargs):
        if isinstance(items, list):
            if self.runsafe and any((not isinstance(item, np.ndarray) for \
                                                        item in items)):
                raise TypeError('Expected npndarrays')
            for item in items:
                self._mutate(item, self.get_mutrate(),\
                             kargs.get('minv'), kargs.get('maxv'),\
                             kargs.get('isint'))
        elif isinstance(items, np.ndarray):
            self._mutate(items.to_numpy(make_copy=False), self.get_mutrate(),\
                         kargs.get('minv'), kargs.get('maxv'),\
                         kargs.get('isint'))
            raise TypeError

    def _determine_parameters(self, item, **kargs):
        # Verify we have a basicChromosome or descendent of
        if isinstance(item, basicIndividual):
            item = item.get_chromo(make_copy=False)
        elif not isinstance(item, basicChromosome):
            raise TypeError('Expected basicChromosome')

        # Get appropriate parameters
        if kargs is not None:
            kargs = kargs.copy()
        else:
            kargs = {}

        maxv = item.get_maxv()
        if kargs.setdefault('maxv', maxv) != maxv:
            raise ValueError('Given maxv that does not match chromosome')

        minv = item.get_minv()
        if kargs.setdefault('minv', minv) != minv:
            raise ValueError('Given minv that does not match chromosome')

        isint = item.get_dtype() == int
        if kargs.setdefault('isint', isint) != isint:
            raise ValueError('Given isint that does not match chromosome')

        return kargs


    # Applies mutation directly to an individual
    def mutate_chromo(self, item, **kargs):

        if isinstance(item, np.ndarray):
            # Verify we have essential arguments
            for argn in ('maxv', 'minv', 'isint'):
                if argn not in kargs:
                    raise MissingValue\
                            (f'Need {argn} if passing ndarray to mutate_chromo')

            # Apply mutation
            self._mutate_chromo(item, **kargs)
            return

        else:
            # Determines parameters
            kargs = self._determine_parameters(item, **kargs)

            # Apply mutation
            self._mutate_chromo(item.to_numpy(make_copy=False), **kargs)
            return

    # Applies mutations directly to individuals
    def mutate_batch(self, batch, **kargs):
        if self.runsafe:
            if not isinstance(batch, list):
                raise TypeError('Expected list of indvs (not a list)')
            if not all([isinstance(indv, (np.ndarray, basicChromosome, \
                                        basicIndividual)) for indv in batch]):
                raise TypeError('Expected list of indvs, chromos, or ndarrays '+\
                                            '(at least one item wasn\'t)')

        # Determines parameters based off the first individual
        kargs = self._determine_parameters(batch[0], **kargs)
        mutrate = self.get_mutrate()
        minv = kargs.get('minv')
        maxv = kargs.get('maxv')
        isint = kargs.get('isint')

        # Apply mutation to each individual
        for indv in [indv.to_numpy(make_copy=False) \
                        if isinstance(indv, (basicChromosome, basicIndividual)) \
                            else indv \
                        for indv in batch]:
            self._mutate(indv, mutrate, minv, maxv, isint)
