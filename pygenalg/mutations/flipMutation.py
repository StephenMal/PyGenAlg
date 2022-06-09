from .basics.basicMutation import basicMutation
from ..indvs.basics import basicChromosome, basicIndividual
import numpy as np
import random, sys

try:
    import numba as nb
except:
    pass

class uniformMutation(basicMutation):

    __slots__ = ()

    use_nbjit_fxn = 'numba' in sys.modules
    DEFAULT_mutrate = 0.015
    def __init__(self, **kargs):

        super().__init__(**kargs)

        if self.__class__ == uniformMutation:
            self.set_params(**kargs)

    if 'numba' in sys.modules:
        @staticmethod
        @nb.jit(nopython=True, parallel=True)
        def _mutate(chromo, mutrate):
            for indx in nb.prange(len(chromo)):
                if np.random.rand() < mutrate:
                    if chromo[indx] == 0:
                        chromo[indx] = 1
                    else:
                        chromo[indx] = 0
    else:
        @staticmethod
        def _mutate(chromo, mutrate):
            mask = self.nprng.choice([0,1], \
                                     size=len(chromo),\
                                     p=(1-mutrate, mutrate))
            chromo = (chromo + mask) % 2
