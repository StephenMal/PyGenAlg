from .basics.basicChromosome import basicChromosome
from .basics.basicIndividual import basicIndividual
from ..exceptions import *
import numpy as np
import sys
try:
    import numba as nb
except:
    pass

class binaryChromosome(basicChromosome):

    __slots__ = ()

    DEFAULT_length = None
    DEFAULT_maxv = 1
    DEFAULT_minv = 0
    DEFAULT_dtype = int
    DEFAULT_dsize = 8
    DEFAULT_np_dtype = np.uint8
    DEFAULT_minlen = None
    DEFAULT_maxlen = None
    DEFAULT_varlen = False

    def __init__(self, **kargs):
        super().__init__(**kargs)

        if self.__class__ == binaryChromosome:
            self.set_params(**kargs)

    def set_dtype(self, dtype):
        if self.runsafe is True and not dtype is int:
            self.pga_warning('dtype must be int for binary chromosome')
        self.dtype = int
    def set_dsize(self, dsize):
        if self.runsafe is True and not dsize == 8:
            self.pga_warning('dsize must be 8 for binary chromosome')
        self.dsize = 8
    def set_minv(self, minv):
        if self.runsafe is True and minv != 0:
            self.pga_warning('min must be 0 for binary chromosome')
        self.minv = 0
    def set_maxv(self, maxv):
        if self.runsafe is True and maxv != 1:
            self.pga_warning('max must be 1 for binary chromosome')
        self.maxv = 1
    def determine_npdtype(self, *args, **kargs):
        self.np_dtype = np.uint8

    def set_params(self, **kargs):
        kargs = kargs.copy()
        kargs.update({'max':1, 'min':0, 'dtype':int})
        super().set_params(**kargs)

    if 'numba' in sys.modules:
        @staticmethod
        @nb.jit(nopython=True, parallel=True)
        def _min(vals):
            for indx in nb.prange(len(vals)):
                if vals[indx] == 0:
                    return 0
            return 1
        @staticmethod
        @nb.jit(nopython=True, parallel=True)
        def _max(vals):
            for indx in nb.prange(len(vals)):
                if vals[indx] == 1:
                    return 1
            return 0
        @staticmethod
        @nb.jit(nopython=True)
        def _get(vals, indx):
            if vals[indx] == 0:
                return 0
            return 1

        def __getitem__(self, indx):
            return self._get(self.vals, indx)

        @staticmethod
        @nb.jit(nopython=True)
        def _hash(arr):
            if len(arr) < 10000:
                return np.sum(np.arange(1,len(arr)+1)*arr)
            else:
                rng = np.random.RandomState(89)
                inds = rng.randint(low=0, high=a.size, size=10000)
                b = a.flat[inds]
                b.flags.writeable = False
                return hash(b.data)

        def __hash__(self):
            return self._hash(self.vals)
    else:
        @staticmethod
        def _min(vals):
            return 0 if 0 in vals else 1
        @staticmethod
        def _max(vals):
            return 1 if 1 in vals else 0
        def __getitem__(self, indx):
            return 0 if self.vals.__getitem__(indx) == 0 else 1

        def __hash__(self):
            arr = self.vals
            return np.sum(\
                np.logspace(1,len(arr),num=len(arr)-2+1,base=10,dtype='int')*arr)

        @staticmethod
        def _search_npsubarr(arr, subarr):
            # Get lengths
            arrlen = len(arr)
            sublen = len(subarr)
            # Return empty if sublen is longer than arrlen
            if sublen > arrlen:
                return
            # Check if in the beginning
            beginning = True
            for j in range(sublen):
                if arr[j] != subarr[j]:
                    beginning = False
            if beginning:
                yield 0

            subsum = sum(subarr)
            cursum = sum(arr[:sublen])

            for i in range(1, arrlen - sublen + 1):
                subsum += arr[i + sublen - 1] - arr[i - 1]
                if subsum == cursum:
                    match_found = True
                    for j in range(sublen):
                        if arr[i + j] != subarr[j]:
                            match_found = False
                    if match_found == True:
                        yield i
        @staticmethod
        def _search_mult_npsubarr(arr, subarrs):
            # Get length of array
            arrlen = len(arr)
            # Get length of subarrays
            sublens = {len(subarr) for subarr in subarrs}
            # Get number of sub arrays
            n_subarrs = len(subarrs)

            # Return empty if sublen is longer than arrlen
            if min(sublen) > arrlen:
                return

            subs = {}
            for subarr in subarrs:
                subarrsum = sum(subarr)
                lst = subs.setdefault(subarrsum, [])
                lst.append(subarr)

            # Get hash of subarr
            subsums = [sum(subarr) for subarr in subarrs]

            # Get hash of arr
            cursums = {length:sum(arr[:length]) for length in sublens}

            for length, cursum in cursums.items():
                if cursum in subs:
                    for sub in subs[curhash]:
                        if len(sub) == length:
                            for i in range(length):
                                if arr[i] != sub[i]:
                                    break
                            yield 0, sub

            for i in range(1, arrlen - min(sublen) + 1):
                remaining_length = arrlen - i
                for length in cursums.keys():
                    # Skip length if too long
                    if length > remaining_length:
                        continue
                    # Update sum
                    cursums[length] += ([i + sublen - 1] - sums[i - 1])
                    if cursums[length] in subs:
                        for sub in subs[curhash]:
                            if len(sub) == length:
                                for j in range(length):
                                    if arr[i] != sub[i]:
                                        break
                                yield i, sub

class binaryIndividual(basicIndividual):

    __slots__ = 'mtype', 'maxm', 'minm', 'signbit', 'n_genes', 'gene_size'

    chromo_class = binaryChromosome

    DEFAULT_n_genes = None
    DEFAULT_gene_size = None
    DEFAULT_mtype = None
    DEFAULT_maxm = None
    DEFAULT_minm = None
    DEFAULT_signbit = None

    def __init__(self, **kargs):

        # Gene information
        self.n_genes , self.gene_size = None, None

        # Mapping information
        self.mtype, self.maxm, self.minm, self.signbit = None, None, None, None

        # Gene information
        self.n_genes, self.gene_size = None, None


        super().__init__(**kargs)

        if self.__class__ == binaryIndividual:
            self.set_params(**kargs)

    def set_mtype(self, mtype):
        if mtype not in (int, float):
            raise ValueError('mtype should be int or float')
        self.mtype = mtype

    def set_maxm(self, maxm):
        if self.mtype is not None and not isinstance(maxm, self.mtype):
            raise TypeError('maxm should be mtype')
        if self.mtype is int:
            self.pga_warning('max does not affect binary indv when mapped to ints')
        self.maxm = maxm

    def set_minm(self, minm):
        if self.mtype is not None and not isinstance(minm, self.mtype):
            raise TypeError('minm should be mtype')
        if self.mtype is int:
            self.pga_warning('min does not affect binary indv when mapped to ints')
        self.minm = minm

    def set_signbit(self, signbit):
        if not isinstance(signbit, bool):
            raise TypeError('signbit parameter should be a bool')
        if self.mtype is not None and self.mtype is float:
            self.pga_warning('Signbit is irrelevant when mapped to float')
        self.signbit = signbit

    def set_n_genes(self, n_genes):
        if not isinstance(n_genes, int):
            raise TypeError('n_genes should be int')
        if n_genes < 1:
            raise ValueError('n_genes should be greater than 0')
        self.n_genes = n_genes

    def set_gene_size(self, gene_size):
        if not isinstance(gene_size, int):
            raise TypeError('gene_size should be an int')
        if gene_size < 1:
            raise ValueError('gene_size should greater than 0')
        self.gene_size = gene_size

    def set_params(self, **kargs):

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

        if 'n_genes' in kargs:
            self.set_n_genes(kargs.get('n_genes'))
        elif self.n_genes is None and ('n_genes' in self.config or \
                                        self.DEFAULT_n_genes is not None):
            self.set_n_genes(self.config.get('n_genes',self.DEFAULT_n_genes,\
                                                        dtype=int, mineq=1))

        if 'gene_size' in kargs:
            self.set_gene_size(kargs.get('gene_size'))
        elif self.gene_size is None and ('gene_size' in self.config or \
                                         self.DEFAULT_gene_size is not None):
            self.set_gene_size(self.config.get('gene_size', self.DEFAULT_gene_size,\
                                                    dtype=int, mineq=1))

        if 'signbit' in kargs:
            self.set_signbit(kargs.get('signbit'))
        elif self.signbit is None and ('signbit' in self.config or \
                                         self.DEFAULT_signbit is not None):
            self.set_signbit(self.config.get('signbit',self.DEFAULT_signbit,\
                                                dtype=bool))

        # Create copy of kargs not including data that shouldn't be passed up
        kargs = {argname:arg for argname,arg in kargs.items() \
                    if argname not in ('max', 'min', 'dtype')}

        super().set_params(**kargs)

    ''' Overwrite key get functions to make more sense with mapping '''
    def get_dtype(self):
        return self.mtype

    def get_max(self):
        return self.maxm

    def get_min(self):
        return self.minm

    ''' Mapping function '''
    # Set up mapping methods, numba and nonnumba versions
    if 'numba' in sys.modules:
        @staticmethod
        @nb.jit(nopython=True, parallel=True)
        def map(vals, n_genes, is_int, use_signbit, minm, maxm):
            split = np.split(vals, n_genes)
            nbits = len(split[0])
            res = np.zeros(n_genes)
            if is_int == True:
                if use_signbit == True:
                    mulby = np.append(np.zeros(1),\
                                      np.power(np.ones(nbits-1)*2, \
                                               np.arange(0, nbits-1)))
                    for gene_num in nb.prange(n_genes):
                        spl = split[gene_num]
                        if spl[0] == 0:
                            res[gene_num] = sum(spl * mulby)
                        else:
                            res[gene_num] = -1 * sum(spl * mulby)
                    return res
                else:
                    mulby = np.power(np.ones(nbits)*2, np.arange(0, nbits))
                    for gene_num in mb.prange(n_genes):
                        spl = split[gene_num]
                        res[gene_num] = sum(spl * mulby)
                    return res

            else:
                mulby = np.power(np.ones(nbits)*2, np.arange(0, nbits))
                oldmax = sum(mulby)
                for gene_num in nb.prange(n_genes):
                    spl = split[gene_num]
                    res[gene_num] = sum(spl * mulby)
                return ((res / oldmax) * (maxm - minm)) + minm

    else:
        @staticmethod
        def map(vals, n_genes, is_int, use_signbit, minm, maxm):
            split = np.split(vals, n_genes)
            nbits = len(split[0])
            if is_int == True:
                if use_signbit:
                    mulby = np.array([0] + [2 ** i for i in range(0, nbits-1)])
                    return np.array([splt*mulby if splt[0] == 0 \
                                                else -1*splt*mulby \
                                                            for splt in split])
                else:
                    mulby = np.power(np.ones(nbits)*2, np.arange(0, nbits))
                    mulby = np.array([2 ** i for i in range(0, nbits)])
                    return np.array([splt*mulby for splt in split])
            else:
                mulby = np.array([2 ** i for i in range(0, nbits)])
                oldmax = sum(mulby)
                for gene_num in range(n_genes):
                    spl = split[gene_num]
                    res[gene_num] = -1 * sum(spl * mulby)
                return ((res / oldmax) * (maxm - minm)) + minm


    def get_mapped(self):
        if self.chromo is None:
            raise MissingPackingVal(argname='chromo')

        if self.n_genes is None:
            raise MissingPackingVal(argname='n_genes')

        # Map dtype
        if self.mtype is None:
            raise MissingPackingVal(argname='mtype')

        elif self.mtype is float:
            if self.minm is not None:
                minm = self.minm
            else:
                self.pga_warning('No min is given, assuming 0')
                minm = 0

            if self.maxm is not None:
                maxm = self.maxm
            else:
                self.pga_warning('No max is given, assuming 1')
                maxm = 0
        else:
            minm, maxm = 0, 0

        return self.map(self.to_numpy(make_copy=False), \
                        self.n_genes, \
                        self.mtype == int, \
                        self.signbit, \
                        minm, maxm)

    def pack(self, **kargs):
        dct = super().pack(**kargs)

        if self.chromo is not None:
            dct['chromo'] = self.chromo.pack(**kargs)
        if self.n_genes is None:
            raise MissingPackingVal(argname='n_genes')

        if self.mtype is int:
            if self.signbit is None:
                raise MissingPackingVal(argname='signbit')
            dct.update({'mtype':'int',\
                        'signbit':self.signbit})
            return dct
        elif self.mtype is float:
            if self.minm is None:
                raise MissingPackingVal(argname='minm')
            if self.maxm is None:
                raise MissingPackingVal(argname='maxm')
            dct.update({'mtype':'float',\
                        'minm':self.minm,\
                        'maxm':self.maxm})
            return dct
        elif self.mtype is None:
            raise MissingPackingVal(argname='mtype')
        else:
            raise NotImplementedError('Only supports mtype == float or int')

        return dct

    @classmethod
    def unpack(cls, dct):
        newdct = {}

        # Check for most critical values
        for argname in ('chromo', 'mtype', 'n_genes'):
            if argname not in dct:
                raise MissingPackingVal(argname=argname)
            else:
                newdct[argname] = dct[argname]

        # Optional values
        for argname in ('fit', 'id', 'attrs'):
            if argname in dct:
                newdct[argname] = dct[argname]

        # Create chromosome from chromo dict
        if 'chromo' in newdct:
            newdct['chromo'] = binaryChromosome.unpack(newdct['chromo'])
            newdct['length'] = dct.get('length', len(newdct['chromo']))
        else:
            newdct['length'] = dct.get('length')
        newdct['nbits'] = dct.get('nbits', length / n_genes)

        # Either get or calculate other values
        if newdct['length'] % newdct['nbits'] != 0:
            length, n_genes = newdct['length'], newdct['n_genes']
            raise ValueError(f'length ({length})should be evenly divisible '+
                                                    f'by n_genes ({n_genes})')

        # Fix mtype if string
        if isinstance(newdct['mtype'], str):
            if newdct['mtype'] == 'int':
                newdct['mtype'] = int
            elif newdct['mtype'] == 'float':
                newdct['mtype'] = float
            else:
                raise NotImplementedError('str -> type only supports int/float')

        # Get mtype specific values
        if newdct['mtype'] is int:
            if 'signbit' not in dct:
                raise MissingPackingVal(argname='signbit')
            newdct['signbit'] = cls.int_to_bool(dct.get('signbit'))
        elif mtype is float:
            for argname in ('minm', 'maxm'):
                if argname not in dct:
                    raise MissingPackingVal(argname=argname)
            newdct['minm'], newdct['maxm'] = kargs.get('minm'), kargs.get('maxm')

        return super().unpack(dct)
