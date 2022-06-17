from .basics import basicChromosome, basicIndividual
from ..exceptions import *
import sys

try:
    import numba as nb
except:
    pass
class proportionalChromosome(basicChromosome):

    __slots__ = ()

    DEFAULT_length = None
    DEFAULT_maxv = None
    DEFAULT_minv = 0
    DEFAULT_dtype = int
    DEFAULT_dsize = 64
    DEFAULT_np_dtype = None
    DEFAULT_minlen = None
    DEFAULT_maxlen = None
    DEFAULT_varlen = True

    def __init__(self, **kargs):
        super().__init__(**kargs)

        if self.__class__ == proportionalChromosome:
            self.set_params(**kargs)

    def __hash__(self):
        (unique, counts) = numpy.unique(number_list, return_counts=True)
        return hash(frozenset((int(val),int(cnt)) for val, cnt in zip(unique, counts)))

    def __eq__(self):
        if isinstance(other, np.ndarray):
            other_arr = other
        elif isinstance(other, (basicChromosome, basicIndividual)):
            other_arr = other.to_numpy(make_copy=False)

        # If not the same length, their frequencies wont add the same
        if len(other_arr) != len(self.vals):
            return False

        # Get unique count and their frequencies
        other_unique, other_ct = np.unique(other_arr, return_counts=True)
        self_unique, self_ct = np.unique(self.vals, return_counts=True)

        # If not the same number of unique values, return False
        if len(other_unique) != len(self_unique):
            return False
        # If the unique values or counts are off return False
        if other_unique != self_unique or other_ct != self_ct:
            return False
        # Otherwise they are equal
        return True

    ''' Parameters '''

    def set_minv(self, num):
        if num != 0:
            raise ValueError('minv must be 0 for Proportional GA')
        super().set_minv(num)

    def set_params(self, **kargs):

        kargs = kargs.copy()

        if 'n_chars' in kargs:
            n_chars = kargs.get('n_chars')
        elif 'n_chars' in self.config:
            n_chars = self.config.get('n_chars', dtype=int, mineq=1)
        else:
            raise MissingValue('Need n_chars')

        kargs.update({'min':0, 'max':n_chars, 'dtype':int})

        super().set_params(**kargs)

    def max(self):
        return  np.bincount(self.vals).argmax()

    def min(self):
        return  np.bincount(self.vals).argmin()



class proportionalIndividual(basicIndividual):

    __slots__ = ('n_genes', 'num_noncoding', 'mapfxn',\
                 'mtype', 'maxm', 'minm')

    chromo_class = proportionalChromosome

    DEFAULT_mtype = None
    DEFAULT_maxm = None
    DEFAULT_minm = None
    DEFAULT_mapfxn = 2
    DEFAULT_noncoding = 0
    DEFAULT_n_genes = None

    def __init__(self, **kargs):

        self.n_genes = None
        self.num_noncoding = None

        self.mapfxn = None
        self.mtype, self.maxm, self.minm = None, None, None


        super().__init__(**kargs)

        if self.__class__ == proportionalIndividual:
            self.set_params(**kargs)



    if 'numba' in sys.modules:
        @staticmethod
        @nb.jit(nopython=True)
        def _pga1_map(arr, minv, maxv, n_genes, n_noncoding):
            # Determine the range (for scaling)
            range = maxv - minv
            # Count the numbers' frequencies
            freqs = np.bincount(arr, minlength=n_genes*2)
            # Create a return array
            ret = np.zeros(n_genes)
            # For each unique number, place the count in its respective index
            for chr in range(n_genes):
                # Only get the first n_genes characters (rest are noncoding)
                if freqs[chr] < n_genes:
                    ret[chr] = (count[chr] * range) + minv
            return ret
        @staticmethod
        @nb.jit(nopython=True)
        def _pga2_map(arr, minv, maxv, n_genes, n_noncoding):
            # Determine the range (for scaling)
            range = maxv - minv
            # Count the numbers' frequencies
            freqs = np.bincount(arr, minlength=n_genes*2)
            # Make return array
            ret = np.zeros(n_genes)
            # Map the quantities
            for indx in range(len(n_genes)):
                pos = indx*2
                neg = (indx*2)+1
                # If positive character is 0, just skip
                if freqs[pos] == 0:
                    ret[indx] = minv
                    continue
                ret[indx] = ((freqs[pos] / (freqs[pos]+freqs[neg]))*range) + minv
            return ret
        @staticmethod
        @nb.jit(nopython=True)
        def _pga3_map(arr, minv, maxv, n_genes, n_noncoding):
            # Determine the range (for scaling)
            range = maxv - minv
            # Count the numbers' frequencies
            freqs = np.bincount(arr, minlength=n_genes*2)
            # Make return array
            ret = np.zeros(n_genes)
            # Map the quantities
            for indx in range(len(n_genes)):
                pos = indx*2
                neg = (indx*2)+1
                # Special conditions if pos == neg
                if freqs[pos] == freqs[neg]:
                    # If both are 0, set at minimum
                    if freqs[pos] == 0 and freqs[neg] == 0:
                        ret[indx] = minv
                    else: # Otherwise, make maxv
                        ret[indx] = maxv
                elif freqs[pos] == 0 or freqs[neg] == 0:
                    # If either value is zero, skip the math and make it minv
                    ret[indx] = minv
                elif freqs[pos] < freqs[neg]:
                    ret[indx] = minv + ((freqs[pos]/freqs[neg])*range)
                else:
                    ret[indx] = minv + ((freqs[neg]/freqs[pos])*range)

            return ret
    else:
        raise Exception

    def get_mapped(self):
        if self.mapfxn is None:
            raise MissingValue('No mapfxn provided')
        # Apply pga map fxn
        map = self.mapfxn(self.chromo.to_numpy(make_copy=False), self.minm, \
                                    self.maxm, self.n_genes, self.n_noncoding)
        # Round the values to the nearest int for int
        if self.mtype is int:
            return map.round()
        return map


    ''' Parameters '''

    def set_n_genes(self, n_genes):
        if not isinstance(n_genes, int):
            if isinstance(n_genes, str) or \
                    (isinstance(n_genes, float) and n_genes.is_integer()):
                n_genes = int(n_genes)
            else:
                raise TypeError('Was expecting integer')
        self.n_genes = n_genes

    def set_num_noncoding(self, n):
        if not isinstance(n, int):
            if isinstance(n, str) or (isinstance(n, float) and n.is_integer()):
                n = int(n)
            else:
                raise TypeError('Was expecting integer')
        self.num_noncoding = n

    def set_mapfxn(self, mapfxn):

        if not isinstance(mapfxn, int) or callable(mapfxn):
            if isinstance(mapfxn, str) or \
                        (isinstance(mapfxn, float) and mapfxn.is_integer()):
                mapfxn = int(mapfxn)

        if isinstance(mapfxn, int):
            if mapfxn == 1:
                self.mapfxn = self._pga1_map
            elif mapfxn == 2:
                self.mapfxn = self._pga2_map
            elif mapfxn == 3:
                self.mapfxn = self._pga3_map
        elif callable(mapfxn):
            self.mapfxn = mapfxn
        else:
            raise TypeError('Expected int (1,2,3) or callable mapfxn')

    def set_mtype(self, mtype):
        if isinstance(mtype, str):
            if mtype == 'int':
                mtype = int
            elif mtype == 'float':
                mtype = float
        if mtype not in (int, float):
            raise ValueError('mtype should be int or float')
        self.mtype = mtype

    def set_maxm(self, maxm):
        if self.mtype is not None and not isinstance(maxm, self.mtype):
            raise TypeError('maxm should be mtype')
        self.maxm = maxm

    def set_minm(self, minm):
        if self.mtype is not None and not isinstance(minm, self.mtype):
            raise TypeError('minm should be mtype')
        self.minm = minm

    def set_params(self, **kargs):

        if 'n_genes' in kargs:
            self.set_n_genes(kargs.get('n_genes'))
        elif self.n_genes is None and ('n_genes' in self.config or \
                                        self.DEFAULT_n_genes is not None):
            self.set_n_genes(self.config.get('n_genes', self.DEFAULT_n_genes, \
                                                            dtype=int, mineq=1))

        if 'num_noncoding' in kargs:
            self.set_num_noncoding(kargs.get('num_noncoding'))
        elif self.num_noncoding is None and ('num_noncoding' in self.config or \
                                            self.DEFAULT_noncoding is not None):
            self.set_num_noncoding(self.config.get('num_noncoding', \
                                                self.DEFAULT_noncoding,\
                                                dtype=int, mineq=0))

        if 'mapfxn' in kargs:
            self.set_mapfxn(kargs.get('mapfxn'))
        elif self.mapfxn is None and ('mapfxn' in self.config or \
                                            self.DEFAULT_mapfxn is not None):
            self.set_mapfxn(self.config.get('mapfxn', self.DEFAULT_mapfxn))

        n_chars = self.n_genes
        if self.mapfxn == 2 or self.mapfxn == 3:
            n_chars = n_chars * 2
        if self.num_noncoding is not None:
            n_chars = n_chars + self.num_noncoding
        cfg_chars = self.config.get('n_chars', n_chars)
        if cfg_chars != n_chars:
            raise ValueError(f'Need {n_chars} characters but is set at {cfg_chars}')

        if 'mtype' in kargs:
            self.set_mtype(kargs.get('mtype'))
        elif 'dtype' in kargs:
            self.set_mtype(kargs.get('dtype'))
        elif self.mtype is None:
            if 'mtype' in self.config:
                self.set_mtype(self.config.get('mtype', options=(int, float)))
            elif 'dtype' in self.config:
                self.set_mtype(self.config.get('dtype', options=(int, float)))
            elif self.DEFAULT_mtype is not None:
                self.set_mtype(self.config.get('mtype', self.DEFAULT_mtype, \
                                            options=(int, float)))

        if 'max' in kargs:
            self.set_maxm(kargs.get('max'))
            if self.mtype is int:
                self.log.warn('max/min is not relevant for int')
        elif self.maxm is None and ('max' in self.config or \
                                    self.DEFAULT_maxm is not None):
            self.set_maxm(self.config.get('max', self.DEFAULT_maxm, \
                                                        dtype=(float,int)))

        if 'min' in kargs:
            self.set_minm(kargs.get('min'))
            if self.mtype is int:
                self.log.warn('max/min is not relevant for int')
        elif self.minm is None and ('min' in self.config or \
                                    self.DEFAULT_minm is not None):
            self.set_minm(self.config.get('min', self.DEFAULT_minm, \
                                                            dtype=(float,int)))

        if (self.mtype is int or self.mtype is None) and \
                                self.maxm is not None and self.minm is not None:
            if self.maxm <= self.minm:
                raise ValueError('maxm needs to be greater than minm')
            if self.minm >= self.maxm:
                raise ValueError('minm needs to be greater than maxm')

        super().set_params(**kargs)

    def get_dtype(self):
        return self.mtype

    def get_max(self):
        return self.maxm

    def get_min(self):
        return self.minm
