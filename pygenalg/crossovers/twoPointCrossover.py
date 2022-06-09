from .basics.basicCrossover import basicCrossover
import numpy as np
import sys
try:
    import numba as nb
except:
    pass


class twoPointCrossover(basicCrossover):

    __slots__ = ()

    DEFAULT_xovrate = 0.9
    DEFAULT_n_parents = 2
    DEFAULT_varlen_okay = False

    def __init__(self, **kargs):
        super().__init__(**kargs)
        if self.__class__ == twoPointCrossover:
            self.set_params(**kargs)

    if 'numba' in sys.modules:
        @staticmethod
        @nb.jit(nopython=True)
        def _cross(chromo1, chromoe2):
            split_pt1 = np.random.randint(0, len(chromos0))
            split_pt2 = np.random.randint(0, len(chromos0))

            if split_pt1 > split_pt2:
                split_pt1, split_pt2 = split_pt2, split_pt1
            elif split_pt1 == split_pt2:
                return [np.append(chromos1[:split_pt1], chromos1[split_pt1:]),\
                        np.append(chromos2[:split_pt1], chromos2[split_pt1:])]

            return [np.concatenate((chromos1[:split_pt1], \
                                    chromos2[split_pt1:split_pt2], \
                                    chromos1[split_pt2:])), \
                    np.concatenate((chromos2[:split_pt1], \
                                    chromos1[split_pt1:split_pt2], \
                                    chromos2[split_pt2:]))]
    else:
        @staticmethod
        def _cross(chromo1, chromo2):

            split_pt1 = random.randint(0, len(chromos0))
            split_pt2 = random.randint(0, len(chromos0))

            if split_pt1 > split_pt2:
                split_pts2 = split_pts1
            elif split_pt1 == split_pt2:
                return [np.append(chromos1[:split_pt1], chromos2[split_pt1:]),\
                        np.append(chromos2[:split_pt1], chromos1[split_pt1:])]

            return [np.concatenate((chromos1[:split_pt1], \
                                    chromos2[split_pt1:split_pt2], \
                                    chromos1[split_pt2:])), \
                    np.concatenate((chromos2[:split_pt1], \
                                    chromos1[split_pt1:split_pt2], \
                                    chromos2[split_pt2:]))]
