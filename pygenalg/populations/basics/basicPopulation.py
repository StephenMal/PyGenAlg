from ...basics import basicComponent
from ...indvs.basics import basicChromosome, basicIndividual
from ...exceptions import *
import pygenalg.indvs
import inspect, sys
from statistics import mean, stdev
from copy import deepcopy
from collections.abc import MutableMapping
import numpy as np
import math

try:
    from scipy.spatial import distance_matrix
except:
    pass

try:
    import numba as nb
except:
    pass

class basicPopulation(basicComponent):

    __slots__ = ('poplst', 'rep', 'popsize', 'maxsize', 'minsize', 'varsize')

    def __init__(self, **kargs):
        # Initialize all values to None
        self.poplst, self.rep,\
            self.popsize, self.maxsize, self.minsize, self.varsize = \
                None, None, None, None, None, None

        super().__init__(**kargs)

        # If making this class, set parameters
        if self.__class__ == basicPopulation:
            self.set_params(**kargs)

    def __del__(self):
        if self.poplst is not None:
            del self.poplst
        super().__del__()

    ''' Access '''

    def __getitem__(self, indx):
        try:
            return self.poplst.__getitem__(indx)
        except Exception as e:
            if self.poplst is None:
                raise MissingValue('No poplst')
            else:
                raise e

    def __setitem__(self, indx, item):
        if not isinstance(item, basicIndividual):
            if isinstance(item, basicChromosome):
                raise NotImplementedError('Cannot handle accepting chromosome yet')
            raise TypeError('Expected an individual')
        try:
            self.poplst.__setitem__(indx, item)
        except Exception as e:
            if self.poplst is None:
                raise MissingValue('No poplst')
            else:
                raise e

    def append(self, item):
        if self.varsize:
            if len(self.poplst) + 1 > self.maxsize:
                raise ValueError("Would exceed maximum length")
            self.poplst.append(items)
        else:
            raise ValueError('Cannot extend to fix length')

    def extend(self, items):
        if self.varsize:
            if len(self.poplst) + len(items) > self.maxsize:
                raise ValueError("Would exceed maximum length")
            self.poplst.extend(items)
        else:
            raise ValueError('Cannot extend to fix length')

    def insert(self, indx, item):
        if self.varsize:
            if len(self.poplst) + 1 > self.maxsize:
                raise ValueError("Would exceed maximum length")
            self.poplst.insert(indx, items)
        else:
            raise ValueError('Cannot extend to fix length')

    def pop(self, indx):
        return self.poplst.pop(indx)

    def __iter__(self):
        if self.poplst is None:
            return iter([])
        return self.poplst.__iter__()

    def __len__(self):
        if self.poplst is None:
            return 0
        return self.__len__(self.poplst)

    ''' Analytics '''

    def sort(self, **kargs):
        self.poplst.sort(**kargs)

    def mean(self, z=None):
        if z is None:
            return mean((indv.get_fit() for indv in self.poplst))
        elif z is not None:
            fits = [indv.get_fit() for indv in self.poplst]
            avg, std = mean(fits), stdev(fits)
            return avg, z*(std/math.sqrt(len(fits)))

    def stdev(self):
        return stdev((indv.get_fit() for indv in self.poplst))

    def nstdev(self):
        lst = [indv.get_fit() for indv in self.poplst]
        meanv = mean(lst)
        # If meanv is 0, we cannot devide stdev by it, return infinity
        if meanv == 0:
            return float('inf')
        return stdev(lst) / meanv

    def max_indv(self):
        max_indv, max_fit = None, float('-inf')
        for indv in self.poplst:
            if indv.get_fit() > max_fit:
                max_indv, max_fit = indv, indv.get_fit()
        return max_indv

    def max(self):
        return max((indv.get_fit() for indv in self.poplst))

    def min_indv(self):
        min_indv, min_fit = None, float('inf')
        for indv in self.poplst:
            if indv.get_fit() < min_fit:
                min_indv, min_fit = indv, indv.get_fit()
        return min_indv

    def min(self):
        return min((indv.get_fit() for indv in self.poplst))

    def consolidate_attrs(self):
        ''' Consolidates the attribute dictionaries '''
        consolidated, keys = dict(), set()
        for n, attr in enumerate((indv.get_attrs() for indv in self.poplst)):
            keys.update(attr.keys())
            for key in keys:
                if key in consolidated:
                    consolidated[key].append(attr.get(key))
                else:
                    consolidated.setdefault(key, [None]*n).append(attr.get(key))
        return consolidated

    ''' Distance '''
    if 'scipy' in sys.modules:
        def get_distance(self):
            return squareform(pdist((indv.get_mapped() for indv in pop), \
                                            metric='euclidean'))
    elif 'numba' in sys.modules:
        @staticmethod
        #@nb.jit(nopython=False)
        def _get_distance(mat):
            results = np.zeros((len(mat), len(mat)))
            for i in range(len(mat)):
                for j in range(len(mat)):
                    results[i][j] = np.linalg.norm(mat[i]-mat[j])
            return results
        def get_distance(self):
            return self._get_distance(np.array([indv.get_mapped() for indv in self.poplst]))
    else:
        def get_distance(self):
            mat = np.array([indv.get_mapped() for indv in pop])
            results = np.zeros(mat)
            for i in range(len(mat)):
                for j in range(len(mat)):
                    results[i][j] = numpy.linalg.norm(mat[i]-mat[j])
            return results
    def apply_distance(self):
        dists = self.get_distance()
        for i, indv in enumerate(self.poplst):
            indv.set_attr('avg_map_dist', mean(dists[i]))


    ''' Generation '''

    def generate(self, *args, **kargs):
        if 'size' in kargs:
            size = kargs.get('size')
        elif self.get_varsize() is True:
            size = random.randint(self.get_minsize(), self.get_maxsize())
        else:
            size = self.get_popsize()

        rep = kargs.get('rep', self.rep)

        if rep is None:
            raise MissingValue('Rep is missing')

        config, log, runsafe = self.config, self.log, self.runsafe
        self.poplst = [rep(config=config,\
                           log=log,\
                           runsafe=runsafe) for x in range(size)]

        if kargs.get('generate_indvs', True):
            for indv in self.poplst:
                indv.generate()

    ''' Parameters '''

    def get_poplst(self):
        if self.poplst is None:
            return []
        return self.poplst

    def to_list(self):
        return self.get_poplst()

    def get_rep(self):
        return self.rep

    def get_popsize(self):
        if self.varsize is False:
            return self.popsize
        else:
            return len(self.get_poplst)

    def get_maxsize(self):
        if self.varsize is False:
            if self.maxsize is not None:
                return self.maxsize
            elif self.popsize is not None:
                return self.popsize
            elif self.minsize is not None:
                return self.minsize
        else:
            if self.maxsize is None:
                raise MissingValue('Missing maxsize')

    def get_minsize(self):
        if self.varsize is False:
            if self.minsize is not None:
                return self.minsize
            elif self.popsize is not None:
                return self.popsize
            elif self.minsize is not None:
                return self.minsize
        else:
            if self.minsize is None:
                raise MissingValue('Missing minsize')

    def get_varsize(self):
        if self.varsize is None:
            if self.minsize is not None and self.maxsize is not None:
                return self.minsize == self.maxsize
            else:
                raise MissingValue('varsize is not set')
        return self.varsize

    def set_poplst(self, poplst, save_copy=False):
        if self.runsafe:
            if not isinstance(poplst, list):
                raise TypeError('Expected a list of individuals')
            if not all((isinstance(indv, (dict, basicIndividual)) for indv in poplst)):
                raise TypeError('Expected a list of individuals')
            if self.minsize is not None and len(poplst) < self.minsize:
                raise ValueError(f'Expected a list of at least {self.minsize}')
            if self.maxsize is not None and len(poplst) > self.maxsize:
                raise ValueError(f'Expected a list shorter than {self.maxsize}')
            if self.maxsize is None and self.minsize is None and \
                    self.varsize is False and self.popsize is not None and \
                    len(poplst) != self.popsize:
                raise ValueError(f'Expected a list of length {self.popsize}')

        if any(isinstance(indv, dict) for indv in poplst):
            self.poplst = [self.unpack_component(indv) \
                                        if isinstance(indv, dict) else indv \
                                            for indv in poplst]
        elif save_copy:
            self.poplst = deepcopy(poplst)
        else:
            self.poplst = poplst

    def set_rep(self, rep):
        if isinstance(rep, str):
            rep = rep.lower()
            if 'individual' not in rep:
                rep = rep + 'Individual'
            elif len(rep) >= 10:
                rep = rep[:-10] + 'Individual'
            else:
                raise ValueError('Expected string of potential individual')

            try:
                rep = getattr(sys.modules['pygenalg.indvs'], rep)
            except AttributeError:
                raise ValueError('Invalid individual')

        if not issubclass(rep, basicIndividual):
            raise TypeError('Expected str or child of basicIndividual class')
        self.rep = rep

    def set_popsize(self, popsize):
        if not isinstance(popsize, int):
            if isinstance(popsize, float) and popsize.is_integer():
                popsize = int(popsize)
            else:
                raise TypeError('Expected int for popsize')
            if popsize < 2:
                raise ValueError('Popsize needs to be greater than 2')
            if self.maxsize is not None and popsize > self.maxsize:
                raise ValueError('Popsize should be less than maxsize if set')
            if self.minsize is not None and popsize < self.minsize:
                raise ValueError('Popsize should be greater than minsize if set')
        self.popsize = popsize

    def set_maxsize(self, maxsize):
        if not isinstance(maxsize, int):
            if isinstance(maxsize, float) and maxsize.is_integer():
                maxsize = int(maxsize)
            else:
                raise TypeError('Expected int for maxsize')
            if maxsize < 2:
                raise ValueError('maxsize needs to be greater than 2')
        self.maxsize = maxsize

    def set_minsize(self, minsize):
        if not isinstance(minsize, int):
            if isinstance(minsize, float) and minsize.is_integer():
                minsize = int(minsize)
            else:
                raise TypeError('Expected int for minsize')
            if minsize < 2:
                raise ValueError('Minsize needs to be greater than 2')
        self.minsize = minsize

    def set_varsize(self, varsize):
        if not isinstance(varsize, bool):
            raise TypeError('Varsize should be a boolean')
        self.varsize = varsize

    def set_params(self, **kargs):

        if 'rep' in kargs:
            self.set_rep(kargs.get('rep'))
        elif self.rep is None and 'rep' in self.config:
            self.set_rep(self.config.get('rep', dtype=(str, basicIndividual)))

        if 'varsize' in kargs:
            self.set_varsize(kargs.get('varsize'))
        elif self.varsize is None and 'varsize' in self.config:
            self.set_varsize(self.config.get('varsize', dtype=bool))

        if 'minsize' in kargs:
            self.set_minsize(kargs.get('minsize'))
        elif self.minsize is None and 'minsize' in self.config:
            self.set_minsize(self.config.get('minsize', dtype=int, mineq=2))

        if 'maxsize' in kargs:
            self.set_maxsize(kargs.get('maxsize'))
        elif self.maxsize is None and 'maxsize' in self.config:
            self.set_maxsize(self.config.get('maxsize', dtype=int, mineq=2))

        if 'popsize' in kargs:
            self.set_popsize(kargs.get('popsize'))
        elif self.popsize is None and 'popsize' in self.config:
            self.set_popsize(self.config.get('popsize', dtype=int, mineq=2))


        if 'poplst' in kargs:
            self.set_poplst(kargs.get('poplst'))
        elif self.poplst is None and 'poplst' in self.config:
            raise Exception('Cannot place poplst in config')

        # Verify sizes inputs work
        if self.minsize is not None and self.maxsize is not None:
            if self.minsize > self.maxsize:
                raise ValueError('Minsize should be less than maxsize')
            if self.varsize is False and self.minsize != self.maxsize:
                raise ValueError('minsize must equal maxsize if varsize is False')
        else:
            if self.varsize is True or self.varsize is None:
                raise MissingValue(\
                    'Need minsize and maxsize if varsize is true or not specified')
            elif self.varsize is False and self.popsize is None:
                raise MissingValue('Need popsize if varsize is False')

    ''' Export '''

    def to_df(self):
        if 'pandas' not in sys.modules:
            try:
                import pandas as pd
            except:
                raise ModuleNotFoundError('Need pandas to convert to dataframe')
        return pd.json_normalize([indv.pack() for indv in self.poplst],\
                                                            orient='columns')


    ''' Packing '''
    @staticmethod
    def compress_dict(*dcts, prefix='', sep='.'):

        if len(dcts) == 0:
            return {}

        def _flatten_dict_gen(d, parent_key='', sep=sep):
            for k, v in d.items():
                new_key = f'{parent_key}{sep}{k}' if parent_key else k
                if isinstance(v, MutableMapping):
                    yield from flatten_dict(v, new_key, sep=sep).items()
                else:
                    yield new_key, v

        def flatten_dict(d, parent_key=prefix, sep=sep):
            return dict(_flatten_dict_gen(d, parent_key, sep))

        dcts = [flatten_dict(d, parent_key=prefix, sep=sep) for d in dcts]


        cmprsd, keys = dict(), set()
        for n, d in enumerate(dcts):
            keys.update(d.keys())
            for key in keys:
                item = d.get(key)
                if item is None:
                    pass
                elif inspect.isclass(item):
                    item = item.__name__
                elif isinstance(item, np.ndarray):
                    item = item.tolist()
                if key in cmprsd:
                    cmprsd[key].append(item)
                else:
                    cmprsd.setdefault(key, [None]*n).append(item)

        # Compress where there's non-unique dtypes
        for key, item in cmprsd.items():
            # Verify uniform values
            v = item[0]
            if isinstance(v, list) or any((i != v for i in item)):
                continue
            # If uniform values, place that value there
            cmprsd[key] = v

        return cmprsd

    @staticmethod
    def decompress_dict(dct, prefix='', sep='.', n=None, split=True):

        # Determine an N value if none is provided
        if n is None:
            n = 1
            for k, i in dct.items():
                if isinstance(i, list):
                    n = max(n, len(i))

        # Decompress the values
        d_dct = {}
        if n > 1:
            for k, i in dct.items():
                if not isinstance(i, list):
                    d_dct[k] = [deepcopy(i) for x in range(n)]
                else:
                    d_dct[k] = i
        else:
            for k, i in dct.items():
                if isinstance(i, list) and len(i) == 1:
                    d_dct[k] = i[0]

        # Split
        if n > 1 and split:
            dcts = [{k:item[i] for k,item in d_dct.items()} for i in range(n)]
        else:
            dcts = [d_dct]

        dct_lst = []
        for dct in dcts:
            final = {}
            for k, i in dct.items():
                if sep in k:
                    keys = k.split(sep)
                    cur_d = final
                    for key in keys[:-1]:
                        cur_d = cur_d.setdefault(key, {})
                    cur_d[keys[-1]] = i
                else:
                    final[k] = i
            dct_lst.append(final)

        return dct_lst


    def pack(self, **kargs):

        dct = super().pack(**kargs)

        ''' Population parameters '''
        dct.update({'popsize':self.get_popsize(),
               'rep':self.rep.__name__})

        if self.maxsize is None or self.minsize is None:
            if self.varsize is None or self.varsize is True:
                raise MissingValue('If varsize is unspecified or True, we need'+\
                                    ' the minsize and maxsize')
        elif self.varsize is True or self.varsize is None:
            dct['maxsize'],dct['minsize'] = self.get_maxsize(),self.get_minsize()

        ''' Individual packaging '''
        #   Reduces repetitive values in effort to reduce memory usage
        # Package the individuals
        if kargs.get('compress', True) and self.poplst is not None:
            packed_indvs = \
                self.compress_dict(*[indv.pack(**kargs) for indv in self.poplst])
            dct['packed_poplst'] = packed_indvs
        elif self.poplst is not None:
            dct['poplst'] = [indv.pack(**kargs) for indv in self.poplst]

        return dct

    @classmethod
    def unpack(self, dct):

        if 'rep' not in dct:
            raise MissingPackingVal(argname='rep')

        if 'packed_poplst' in dct:
            if 'popsize' in dct:
                dct['poplst'] = self.decompress_dict(dct['packed_poplst'],\
                                                     n=int(dct['popsize']))
            else:
                dct['poplst'] = self.decompress_dict(dct['packed_poplst'])

        return super().unpack(dct)
