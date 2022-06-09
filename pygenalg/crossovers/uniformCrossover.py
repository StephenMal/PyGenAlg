from .basics.basicCrossover import basicCrossover
import numpy as np
import sys, random
try:
    import numba as nb
except:
    pass


class uniformCrossover(basicCrossover):

    __slots__ = {'uniform_xov_random_dist':\
                    'Boolean for whether or not the distribution of parents '+\
                    'should be equal'}

    DEFAULT_uniform_xov_random_dist = False
    DEFAULT_xovrate = 0.9
    DEFAULT_n_parents = 2
    DEFAULT_varlen_okay = False

    def __init__(self, **kargs):

        self.uniform_xov_random_dist = None

        super().__init__(**kargs)

        if self.__class__ == uniformCrossover:
            self.set_params(**kargs)

    def set_params(self, **kargs):

        if 'uniform_xov_random_dist' in kargs:
            self.set_uniform_xov_random_dist(kargs.get('uniform_xov_random_dist'))
        elif self.uniform_xov_random_dist is None:
            self.set_uniform_xov_random_dist(\
                                    self.config.get('uniform_xov_random_dist', \
                                    DEFAULT_uniform_xov_random_dist, dtype=bool))

        super().set_params(**kargs)

    if 'numba' in sys.modules:
        @staticmethod
        @nb.jit(nopython=True, parallel=True)
        def _cross(chromo1, chromo2):
            # Create the chromosomes
            ch1, ch2 = np.zeros(len(chromo1)), np.zeros(len(chromo1))
            # Go through and place randomly
            for indx in nb.prange(len(chromo1)):
                if np.random.random() < 0.5:
                    ch1[indx] = chromo1[indx]
                    ch2[indx] = chromo2[indx]
                else:
                    ch1[indx] = chromo2[indx]
                    ch2[indx] = chromo1[indx]

            return [ch1, ch2]
    else:
        @staticmethod
        def _cross(chromo1, chromo2):
            # Create masks
            mask = np.random.choices([0,1], size=len(chromo1))
            # Create new chromosomes arrays
            ch1, ch2 = np.zeros(len(chromo1)), np.zeros(len(chromo1))
            # Applying the masks, take only from certain values
            ch1[mask], ch2[mask], ch2[mask], ch1[mask] = \
                                            chromo1, chromo2, chromo1, chromo2

            return [np.append(chromo1[split_pt:], chromo2[:split_pt]),\
                    np.append(chromo2[split_pt:], chromo1[:split_pt])]

    def pack(self, **kargs):

        if self.uniform_xov_random_dist is None and \
                                self.DEFAULT_uniform_xov_random_dist is None:
            raise MissingPackingVal('uniform_xov_random_dist')

        dct = super().pack(**kargs)

        if kargs.get('incl_defs', False) or self.uniform_xov_random_dist != \
                                            self.DEFAULT_uniform_xov_random_dist:
            dct['uniform_xov_random_dist'] = self.uniform_xov_random_dist
