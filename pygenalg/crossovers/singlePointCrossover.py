from .basics.basicCrossover import basicCrossover
import numpy as np
import sys, random
try:
    import numba as nb
except:
    pass


class singlePointCrossover(basicCrossover):

    __slots__ = ()

    DEFAULT_xovrate = 0.9
    DEFAULT_n_parents = 2
    DEFAULT_varlen_okay = False

    def __init__(self, **kargs):
        super().__init__(**kargs)
        if self.__class__ == singlePointCrossover:
            self.set_params(**kargs)

    if 'numba' in sys.modules:
        @staticmethod
        @nb.jit(nopython=True)
        def _cross(chromos):
            split_pt = np.random.randint(0, len(chromos[0]))
            return [np.append(chromos[0][split_pt:], chromos[1][:split_pt]),\
                    np.append(chromos[1][split_pt:], chromos[0][:split_pt])]
    else:
        @staticmethod
        def _cross(chromos):
            split_pt = random.randint(0, len(chromos[0]))
            return [np.append(chromos[0][split_pt:], chromos[1][:split_pt]),\
                    np.append(chromos[1][split_pt:], chromos[0][:split_pt])]
