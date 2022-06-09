from pygenalg.evaluators.basics.basicEvaluator import basicEvaluator
import sys
import numpy as np

try:
    import numba as nb
    from numba import prange
except:
    pass


class logisticRegressionEvaluator(basicEvaluator):

    __slots__ = ('train_feats', 'train_lbls', \
                 'test_feats', 'test_lbls', \
                 'toggles','exponents',\
                 'standardize', 'normalize')

    DEFAULT_normalize = False
    DEFAULT_standardize = False
    DEFAULT_toggles = True
    DEFAULT_maximize = False

    def __init__(self, *args, **kargs):

        super().__init__(*args, **kargs)

        for arg in self.__slots__:
            setattr(self, arg, None)


        if self.__class__ is logisticRegressionEvaluator:
            self.set_params(**kargs)


    if 'numba' in sys.modules:
        @staticmethod
        @nb.jit(nopython=True)
        def bce(yhat, y, apply_sigmoid):
            if apply_sigmoid == True:
                yhat = 1 / (1 + np.exp(-x))
            return -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)).mean()

        @staticmethod
        @nb.jit(nopython=True, parallel=True)
        def chrs_to_weights(arrs, toggles):
            if toggles == True:
                weights = np.zeros((len(arrs), len(arrs[0])//2))
                for arr_indx in prange(len(arrs)):
                    for w_indx in range(len(arrs[0])//2):
                        if arrs[arr_indx][w_indx*2+1] > 0:
                            weights[arr_indx][w_indx] = arrs[arr_indx][w_indx*2]
                return weights
            else:
                return np.copy(arrs)

        @staticmethod
        @nb.jit(nopython=False, parallel=False)
        def get_yhats(mat, w_arrs):
            # Create 2D matrix (# of sets of weights, number of entries)
            yhats = np.zeros((len(w_arrs),len(mat)))
            # Go through each provided w_arrs
            for w_indx in prange(len(w_arrs)):
                # Apply weights and find sums
                for m_indx in range(len(mat)):
                    yhats[w_indx][m_indx] = np.sum(mat[m_indx]*w_arrs[w_indx])
                yhats[w_indx] = 1 / np.exp(-yhats[w_indx])
            return yhats

        @staticmethod
        @nb.jit(nopython=True, parallel=True)
        def get_bces(yhats, ytru):
            bce = np.zeros(len(yhats))
            for i in prange(len(yhats)):
                clipped = np.clip(yhats[i], 1E-9, 0.9999999999999999)
                bce[i] = -(ytru * np.log(clipped) + \
                                    (1 - ytru) * np.log(1 - clipped)).mean()
            return bce

        @staticmethod
        @nb.jit(nopython=True, parallel=True)
        def get_accs(yhats, ytru):
            accs = np.zeros(len(yhats))
            for i in prange(len(yhats)):
                correct = 0
                for indx in range(len(ytru)):
                    if yhats[i][indx] < 0.5 and ytru[indx] == 1:
                        correct = correct + 1
                    elif yhats[i][indx] >= 0.5 and ytru[indx] == 0:
                        correct = correct + 1
                accs[i] = correct / len(ytru)
            return accs

        def evaluate(self, indv):
            indv = [indv]
            self.evaluate_batch(indv)

        def evaluate_batch(self, batch):

            if self.runsafe:
                if self.train_lbls is None:
                    raise MissingValue('Missing train_lbls')
                elif self.train_feats is None:
                    raise MissingValue('Missing train_feats')
            # Get all the chromosomes
            arrs = np.array([indv.to_numpy(make_copy=False) for indv in batch])
            # Get weights from chromosomes as arrays
            weights = self.chrs_to_weights(arrs, self.toggles)
            # Get yhats (predictions)
            yhats = self.get_yhats(self.train_feats, weights)
            # Get bce and acc
            bces = self.get_bces(yhats, self.train_lbls)
            accs = self.get_accs(yhats, self.train_lbls)
            # Track best and worst for gen
            best, worst, best_indv, worst_indv = \
                                    float('inf'), float('-inf'), None, None
            # Apply to individuals
            for indv, bce, acc in zip(batch, bces, accs):
                indv.set_fit(bce)
                indv.set_attr('train_acc', acc)
                # Trakck worst and best
                if bce < best:
                    best, best_indv = bce, indv
                elif bce > worst:
                    worst, worst_indv = bce, indv

            # Set the best/worst seen
            self.set_max_indv(worst_indv)
            self.set_min_indv(best_indv)

            # Run the tests if so
            if self.test_feats is not None and self.test_lbls is not None:
                test_yhats = self.get_yhats(self.test_feats, weights)
                test_bces = self.get_bces(test_yhats, self.test_lbls)
                test_accs = self.get_accs(test_yhats, self.test_lbls)
                # Apply to individuals
                for indv, bce, acc in zip(batch, test_bces, test_accs):
                    indv.set_attr('test_bce', bces)
                    indv.set_attr('test_acc', acc)

    else:
        raise Exception

    ''' Data Preprocessing '''
    @staticmethod
    def standardize_mat(matrix, avg=None, std=None):
        if avg is None:
            avg = np.mean(matrix, axis=0)
        if std is None:
            std = np.std(matrix, axis=0)
        return (matrix - avg) / std

    @staticmethod
    def normalize_mat(matrix, maxv=None, minv=None):
        if maxv is None:
            maxv = np.max(matrix, axis=0)
        if minv is None:
            minv = np.min(matrix, axis=0)
        return (matrix - minv) / (maxv - minv)

    ''' Parameters '''
    def set_standardize(self, val):
        if not isinstance(val, bool):
            if isinstance(val, (int, float)):
                if val == 0:
                    val = False
                elif val == 1:
                    val = True
            else:
                raise TypeError
        self.standardize = val

    def set_normalize(self, val):
        if not isinstance(val, bool):
            if isinstance(val, (int, float)):
                if val == 0:
                    val = False
                elif val == 1:
                    val = True
            else:
                raise TypeError
        self.normalize = val

    def set_toggles(self, val):
        if not isinstance(val, bool):
            if isinstance(val, (int, float)):
                if val == 0:
                    val = False
                elif val == 1:
                    val = True
            else:
                raise TypeError
        self.toggles = val

    def set_feats(self, feats, test_or_train='train'):
        if not isinstance(feats, np.ndarray):
            raise TypeError
        if not feats.ndim == 2:
            raise ValueError(f'Expected 2D array (not {feats.ndim}D)')

        if self.standardize:
            feats = self.standardize_mat(feats)

        if self.normalize:
            feats = self.normalize_mat(feats)

        if test_or_train == 'train':
            self.train_feats = feats
        elif test_or_train == 'test':
            self.test_feats = feats



    def set_lbls(self, lbls, test_or_train):
        if not isinstance(lbls, np.ndarray):
            raise TypeError
        if not lbls.ndim == 1:
            raise ValueError(f'Expected 1D array (not {lbls.ndim}D)')

        if self.standardize:
            lbls = self.standardize_mat(lbls)

        if self.normalize:
            lbls = self.normalize_mat(lbls)

        if test_or_train == 'train':
            self.train_lbls = lbls
        elif test_or_train == 'test':
            self.test_lbls = lbls

    def set_params(self, **kargs):

        if 'standardize' in kargs:
            self.set_standardize(kargs.get('standardize'))
        elif self.standardize is None and ('standardize' in self.config or \
                                          self.DEFAULT_standardize is not None):
            self.set_standardize(self.config.get('standardize', \
                                    self.DEFAULT_standardize, dtype=bool))

        if 'normalize' in kargs:
            self.set_normalize(kargs.get('normalize'))
        elif self.normalize is None and ('normalize' in self.config or \
                                          self.DEFAULT_normalize is not None):
            self.set_normalize(self.config.get('normalize', \
                                    self.DEFAULT_normalize, dtype=bool))

        if 'toggles' in kargs:
            self.set_toggles(kargs.get('toggles'))
        elif self.toggles is None and ('toggles' in self.config or \
                                       self.DEFAULT_toggles is not None):
            self.set_toggles(self.config.get('toggles', self.DEFAULT_toggles,\
                                                                dtype=bool))

        for test_train in ('test', 'train'):
            if f'{test_train}_feats' in kargs:
                self.set_feats(kargs.get(f'{test_train}_feats'), \
                                                    test_or_train=test_train)
            elif f'{test_train}_feats' in self.config:
                self.set_feats(self.config.get(f'{test_train}_feats',\
                                    dtype=np.ndarray), test_or_train=test_train)

            if f'{test_train}_lbls' in kargs:
                self.set_lbls(kargs.get(f'{test_train}_lbls'), \
                                                    test_or_train=test_train)
            elif f'{test_train}_lbls' in self.config:
                self.set_lbls(self.config.get(f'{test_train}_lbls',\
                                    dtype=np.ndarray), test_or_train=test_train)

        super().set_params(**kargs)
