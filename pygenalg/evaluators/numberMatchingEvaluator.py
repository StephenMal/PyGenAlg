from .basics.basicEvaluator import basicEvaluator
import numpy as np
from ..exceptions import *
import sys

try:
    import numba as nb
    from numba import prange
except:
    pass

class numberMatchingEvaluator(basicEvaluator):

    __slots__ = {'match_arr':'The array we are attempting to match', \
                 'dtype':'The data type to create the match_arr'}

    # Overwrites
    DEFAULT_maximize = False
    DEFAULT_dynamic = False

    # New defaults
    DEFAULT_match_arr = None
    DEFAULT_dtype = int

    def __init__(self, **kargs):

        self.match_arr = None

        super().__init__(**kargs)

        if self.__class__ == numberMatchingEvaluator:
            self.set_params(**kargs)

    def set_match_arr(self, arr):
        if not isinstance(arr, np.ndarray):
            if isinstance(arr, list):
                arr = np.array(arr)
            else:
                raise TypeError('Expected numpy array for matching array')

        self.match_arr = arr

    def gen_match_arr(self, arr, **kargs):
        if 'indv' in kargs:
            length, minv, maxv, dtype = None, None, None, None
            length = kargs.get('length', len(indv))
            minv = kargs.get('minv', kargs.get('min'), indv.get_minv())

    def set_params(self, **kargs):
        if 'match_arr' in kargs:
            self.set_match_arr(kargs.get('match_arr'))
        elif self.match_arr is None and ('match_arr' in self.config or \
                                            self.DEFAULT_match_arr is not None):
            self.set_match_arr(self.config.get('match_arr', dtype=(list, np.ndarray)))

        super().set_params(**kargs)

    if 'numba' in sys.modules:
        @staticmethod
        @nb.jit(nopython=True, parallel=True)
        def _evaluate(self, arr, match_arr):
            abssum = 0
            for indx in prange(len(match_arr)):
                abssum = abssum + abs(arr[indx] - match_arr[indx])
            return abssum

        def evaluate(self, indv, **kargs):
            if len(indv) != len(self.match_arr):
                raise ValueError('Indv length should equal given match array')
            fit = self._evaluate(indv.to_numpy(make_copy=False),self.match_arr)
            indv.set_fit(fit)

        @staticmethod
        @nb.jit(nopython=True, parallel=True)
        def _evaluate_batch(mat, match_arr):
            res = np.zeros(len(mat))
            for i in prange(len(mat)):
                abssum = 0.0
                for j in prange(len(match_arr)):
                    abssum += abs(mat[i][j] - match_arr[j])
                res[i] = abssum
            return res

        def evaluate_batch(self, batch, **kargs):
            if self.match_arr is None:
                raise MissingValue('Need to have match array')
            # Get fitnesses
            fits = self._evaluate_batch(\
                np.array([indv.to_numpy(make_copy=False) for indv in batch]),\
                self.match_arr)

            update_minmax = kargs.get('update_minmax', True)
            if update_minmax:
                maxfit, minfit = (None, float('-inf')), (None, float('inf'))

            for indv_num, fit in enumerate(fits.tolist()):
                batch[indv_num].set_fit(fit)

                if update_minmax:
                    if fit > maxfit[1]:
                        maxfit = (batch[indv_num], fit)
                    elif fit < minfit[1]:
                        minfit = (batch[indv_num], fit)

            if update_minmax:
                self.set_max_indv(maxfit[0], fit=maxfit[1])
                self.set_min_indv(minfit[0], fit=minfit[1])

            return


    else:
        def evaluate(self, indv, **kargs):
            if self.match_arr is None:
                raise MissingValue('Need to have match array')
            if len(indv) != len(self.match_arr):
                raise ValueError('Indv length should equal given match array')

            fit = np.sum(np.absolute(\
                                indv.to_numpy(make_copy=False)-self.match_arr))

            if kargs.get('update_minmax', True):
                self.set_max_indv(indv, fit=fit)
                self.set_min_indv(indv, fit=fit)

            return fit

        def evaluate_batch(self, batch, **kargs):
            m_arr = self.match_arr
            if self.match_arr is None:
                raise MissingValue('Need to have match array')

            # Create tuples to store min and max fit seen
            update_minmax = kargs.get('update_minmax', True)
            if update_minmax:
                minfit, maxfit = (None, float('-inf')), (None, float('inf'))

            # Iterate through individuals in the batch
            for indv in batch:
                # Evaluate
                fit = np.sum(np.abs(indv.to_numpy(make_copy=False)-m_arr))
                indv.set_fit(fit)

                # If updating minmax, track max and min
                if update_minmax:
                    if fit > maxfit[1]:
                        maxfit = (indv, fit)
                    elif fit < minfit[1]:
                        minfit = (indv, fit)

            # Update minmax if enabled
            if update_minmax:
                self.set_min_indv(minfit[0], fit=minfit[1])
                self.set_max_indv(maxfit[0], fit=maxfit[1])

            return


    def pack(self, **kargs):

        if self.match_arr is None:
            raise MissingPackingVal('match_arr')

        dct = super().pack()
        if isinstance(self.match_arr, np.ndarray):
            dct['match_arr'] = self.match_arr.tolist()
        elif isinstance(self.match_arr, list):
            dct['match_arr'] = self.match_arr.copy()
        else:
            raise TypeError

        return dct

    @classmethod
    def unpack(cls, dct):
        if 'match_arr' not in dct:
            raise MissingPackingVal('match_arr')
        return super().unpack(dct)
