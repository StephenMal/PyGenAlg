from pygenalg.evaluators.basics.basicEvaluator import basicEvaluator
import sys
import numpy as np

try:
    import numba as nb
    from numba import prange
except:
    pass

try:
    import pandas as pd
except:
    pass


class logisticRegressionEvaluator(basicEvaluator):

    __slots__ = ('train_feats', 'train_lbls', \
                 'test_feats', 'test_lbls', \
                 'toggles','exponents','constant',\
                 'standardize', 'normalize',\
                 'has_constant')

    DEFAULT_normalize = False
    DEFAULT_standardize = False
    DEFAULT_toggles = True
    DEFAULT_maximize = False
    DEFAULT_has_constant = True

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
        def chrs_to_weights(arrs, toggles, has_constant):
            if toggles == True:
                if has_constant:
                    weights = np.zeros((len(arrs), (len(arrs[0])//2)+1))
                else:
                    weights = np.zeros((len(arrs), len(arrs[0])//2))
                for arr_indx in prange(len(arrs)):
                    for w_indx in range(len(arrs[0])//2):
                        if arrs[arr_indx][w_indx*2+1] > 0:
                            weights[arr_indx][w_indx] = arrs[arr_indx][w_indx*2]
                    if has_constant:
                        weights[arr_indx][-1] = arrs[arr_indx][-1]
                return weights
            else:
                return np.copy(arrs)

        @staticmethod
        @nb.jit(nopython=True, parallel=False)
        def get_yhats(mat, w_arrs, has_constant):
            # Create 2D matrix (# of sets of weights, number of entries)
            yhats = np.zeros((len(w_arrs),len(mat)))
            # Go through each provided w_arrs
            for w_indx in prange(len(w_arrs)):
                # Apply weights and find sums
                for m_indx in range(len(mat)):
                    if has_constant:
                        yhats[w_indx][m_indx] = np.sum(mat[m_indx]*w_arrs[w_indx][1:])+w_arrs[w_indx][0]
                    else:
                        yhats[w_indx][m_indx] = np.sum(mat[m_indx]*w_arrs[w_indx])

                yhats[w_indx] = 1 / np.exp(-yhats[w_indx])
            return yhats

        @staticmethod
        @nb.jit(nopython=True, parallel=True)
        def get_bces(yhats, ytru):
            bce = np.zeros(len(yhats))
            for i in prange(len(yhats)):
                clipped = np.clip(yhats[i], 1E-9, 1-(1E-9))
                bce[i] = -(ytru * np.log(clipped) + \
                                    (1 - ytru) * np.log(1 - clipped)).mean()
            return bce

        @staticmethod
        @nb.jit(nopython=True, parallel=True)
        def get_accs(yhats, ytru):
            # Accuracy, TP, FP, TN, FN
            accs = np.zeros((len(yhats), 5))
            for i in prange(len(yhats)):
                TP = 0
                FP = 0
                TN = 0
                FN = 0
                correct = 0
                for indx in range(len(ytru)):
                    if yhats[i][indx] < 0.5:
                        if ytru[indx] == 0:
                            correct = correct + 1
                            TN = TN + 1
                        else:
                            FN = FN + 1
                    elif yhats[i][indx] >= 0.5:
                        if ytru[indx] == 1:
                            correct = correct + 1
                            TP = TP + 1
                        else:
                            FP = FP + 1
                accs[i][0] = round(correct / len(ytru), 3)
                accs[i][1] = round(TP / len(ytru), 3)
                accs[i][2] = round(FP / len(ytru), 3)
                accs[i][3] = round(TN / len(ytru), 3)
                accs[i][4] = round(FN / len(ytru), 3)
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
            weights = self.chrs_to_weights(arrs, self.toggles, self.has_constant)

            if self.runsafe and len(weights[0]) != len(self.train_feats[0]) + int(self.has_constant):
                raise ValueError('Number of weights != number of features '+\
                        f'({len(weights[0])} != {len(self.train_feats[0])})')


            # Get yhats (predictions)
            yhats = self.get_yhats(self.train_feats, weights, self.has_constant)
            # Get bce and acc
            bces = self.get_bces(yhats, self.train_lbls)
            accs = self.get_accs(yhats, self.train_lbls)
            # Track best and worst for gen
            best, worst, best_indv, worst_indv = \
                                    float('inf'), float('-inf'), None, None
            # Apply to individuals
            for indv, weight_lst, bce, acc in zip(batch, weights, bces, accs):
                indv.set_fit(bce)
                indv.set_attr('train_acc', acc[0])
                indv.set_attr('train_TP', acc[1])
                indv.set_attr('train_FP', acc[2])
                indv.set_attr('train_TN', acc[3])
                indv.set_attr('train_FN', acc[4])

                # Trakck worst and best
                if bce < best:
                    best, best_indv = bce, indv
                elif bce > worst:
                    worst, worst_indv = bce, indv

                # Set weights in attributes
                indv.set_attr('weights', weight_lst.tolist())

            # Set the best/worst seen
            self.set_max_indv(worst_indv)
            self.set_min_indv(best_indv)

            # Run the tests if so
            if self.test_feats is not None and self.test_lbls is not None:
                test_yhats = self.get_yhats(self.test_feats, weights, self.has_constant)
                test_bces = self.get_bces(test_yhats, self.test_lbls)
                test_accs = self.get_accs(test_yhats, self.test_lbls)
                # Apply to individuals
                for indv, bce, acc in zip(batch, test_bces, test_accs):
                    indv.set_attr('test_bce', bce)
                    indv.set_attr('test_acc', acc[0])
                    indv.set_attr('test_TP', acc[1])
                    indv.set_attr('test_FP', acc[2])
                    indv.set_attr('test_TN', acc[3])
                    indv.set_attr('test_FN', acc[4])

    else:
        raise Exception

    ''' Data Preprocessing '''
    @staticmethod
    def standardize_mat(matrix, avg=None, std=None):
        if avg is None:
            avg = np.mean(matrix, axis=0)
        if std is None:
            std = np.std(matrix, axis=0)
        return np.divide((matrix - avg), std, where = std != 0)

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

    def set_has_constant(self, val):
        if not isinstance(val, bool):
            if isinstance(val, (int, float)):
                if val == 0:
                    val = False
                elif val == 1:
                    val = True
            else:
                raise TypeError
        self.has_constant = val

    def set_feats(self, feats, test_or_train='train'):
        if isinstance(feats, list):
            if not all(isinstance(item, list) for item in feats):
                raise TypeError('Must be 2D list')
        elif isinstance(feats, str):
            if 'pandas' not in sys.modules:
                raise ModuleNotFoundError('Required Pandas if given filepath str')
            if not os.path.exists(feats):
                raise FileNotFoundError(f'{feats} file path was not valid')
            if '.' not in feats:
                raise ValueError(f'No file extension on file ({feats})')
            filetype == feats.split('.')[-1]

            if filetype == 'csv':
                feats = pd.read_csv(feats)
            elif filetype == 'feather':
                feats = pd.read_feather(feats)
            elif filetype == 'parquet':
                feats = pd.read_parquet(feats)

            # Sort
            feats = feats.reindex(sorted(feats.columns), axis=1)
            # Turn into numpy array
            feats = feats.to_numpy()

        elif 'pandas' in sys.modules and isinstance(feats, pd.DataFrame):
            # Sort
            feats = feats.reindex(sorted(feats.columns), axis=1)
            # Turn into numpy array
            feats = feats.to_numpy()
        elif not isinstance(feats, np.ndarray):
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
        if isinstance(lbls, list):
            lbls = lbls.to_numpy()
        elif isinstance(lbls, str):
            if 'pandas' not in sys.modules:
                raise ModuleNotFoundError('Required Pandas if given filepath str')
            if not os.path.exists(feats):
                raise FileNotFoundError(f'{feats} file path was not valid')
            if '.' not in feats:
                raise ValueError(f'No file extension on file ({feats})')
            filetype == feats.split('.')[-1]

            if filetype == 'csv':
                feats = pd.read_csv(feats)
            elif filetype == 'feather':
                feats = pd.read_feather(feats)
            elif filetype == 'parquet':
                feats = pd.read_parquet(feats)
            # Turn into numpy array
            feats = feats.to_numpy()

        elif 'pandas' in sys.modules and isinstance(lbls, pd.Series):
            lbls = np.array(lbls)
        elif not isinstance(lbls, np.ndarray):
            raise TypeError

        if not lbls.ndim == 1:
            raise ValueError(f'Expected 1D array (not {lbls.ndim}D)')

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

        if 'has_constant' in kargs:
            self.set_has_constant(kargs.get('has_constant'))
        elif self.has_constant is None and ('has_constant' in self.config or \
                                            self.DEFAULT_has_constant is not None):
            self.set_has_constant(self.config.get('has_constant', \
                                                    self.DEFAULT_has_constant,\
                                                    dtype=bool))

        for test_train in ('test', 'train'):
            if f'{test_train}_feats' in kargs:
                self.set_feats(kargs.get(f'{test_train}_feats'), \
                                                    test_or_train=test_train)
            elif f'{test_train}_feats' in self.config:
                self.set_feats(self.config.get(f'{test_train}_feats'),\
                                                    test_or_train=test_train)

            if f'{test_train}_lbls' in kargs:
                self.set_lbls(kargs.get(f'{test_train}_lbls'), \
                                                    test_or_train=test_train)
            elif f'{test_train}_lbls' in self.config:
                self.set_lbls(self.config.get(f'{test_train}_lbls'),\
                                                    test_or_train=test_train)

        super().set_params(**kargs)
