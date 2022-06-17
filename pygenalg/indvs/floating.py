from .basics.basicChromosome import basicChromosome
from .basics.basicIndividual import basicIndividual
from ..exceptions import *
import numpy as np
import sys, math, random
from statistics import mean
try:
    import numba as nb
except:
    pass

class floatingChromosome(basicChromosome):

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

        if self.__class__ == floatingChromosome:
            self.set_params(**kargs)

    def set_dtype(self, dtype):
        if self.runsafe is True and not dtype is int:
            self.pga_warning('dtype must be int for floating chromosome')
        self.dtype = int
    def set_dsize(self, dsize):
        if self.runsafe is True and not dsize == 8:
            self.pga_warning('dsize must be 8 for floating chromosome')
        self.dsize = 8
    def set_minv(self, minv):
        if self.runsafe is True and minv != 0:
            self.pga_warning('min must be 0 for floating chromosome')
        self.minv = 0
    def set_maxv(self, maxv):
        if self.runsafe is True and maxv != 1:
            self.pga_warning('max must be 1 for floating chromosome')
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

        @staticmethod
        @nb.jit(nopython=True)
        def float_search(chr, arrs, gene_len):
            # Return nothing if arrs[0] is longer
            if len(arrs[0]) > len(chr):
                return

            # Sum the beginning
            cursum = 0
            for i in range(len(arrs[0])):
                cursum = cursum + chr[i]

            # Stores all the sums of the subarrs
            sums = np.zeros(len(arrs))
            # Iterate through the subarrs
            for indx in range(len(arrs)):
                # Sum it
                sums[indx] = np.sum(arrs[indx])
                # If sum matches the beggining sum, check for matches
                if sums[indx] == cursum:
                    # Assume match is true
                    match = True
                    # Iterate through the subarr
                    for a_indx in range(len(arrs[indx])):
                        # Compare to chromosome, break if not true
                        if arrs[indx][a_indx] != chr[a_indx]:
                            match = False
                            break
                    # If match, yield it.
                    if match == True:
                        yield (indx, len(arrs[indx]), gene_len+len(arrs[indx]))

            # Iterate trhough
            for c_indx in range(1,len(chr)-gene_len):
                # Update the sum
                cursum = cursum - chr[c_indx-1] + chr[c_indx+len(arrs[0])-1]
                # Check if matching sum
                for s_indx in range(len(sums)):
                    # If the sum matches see if its a true match
                    if sums[s_indx] == cursum:
                        match = True # Assume true
                        # Iterate through chr and subarr
                        for a_indx in range(len(arrs[s_indx])):
                            # \/ Verify length doesn't exceed bounds
                            if (a_indx+c_indx) >= len(chr):
                                match = False
                                break
                            elif arrs[s_indx][a_indx] != chr[a_indx+c_indx]:
                                # Check for a match /\
                                match = False
                                break
                        # If a match, yield
                        if match == True:
                            yield (s_indx, \
                                   c_indx+len(arrs[s_indx]), \
                                   c_indx+gene_len+len(arrs[s_indx]))
            return
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

        if 'numba' in sys.modules:
            _search_npsubarr = nb.jit(_search_npsubarr, nopython=True)

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

class floatingIndividual(basicIndividual):

    __slots__ = ('mtype', 'maxm', 'minm', 'signbit', 'n_genes', 'gene_size')

    chromo_class = floatingChromosome

    DEFAULT_n_genes = None
    DEFAULT_gene_size = None
    DEFAULT_mtype = None
    DEFAULT_maxm = None
    DEFAULT_minm = None
    DEFAULT_start_seq = np.array([0,0,0])
    DEFAULT_signbit = None
    DEFAULT_dflt_val = 0

    gene_map = None
    gene_mat = None
    start_seq = DEFAULT_start_seq.copy()
    dflt_val = DEFAULT_dflt_val
    def __init__(self, **kargs):

        # Gene information
        self.n_genes , self.gene_size = None, None

        # Mapping information
        self.mtype, self.maxm, self.minm, self.signbit = None, None, None, None

        # Gene information
        self.n_genes, self.gene_size = None, None

        super().__init__(**kargs)

        if self.__class__ == floatingIndividual:
            self.set_params(**kargs)

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
        if self.mtype is int:
            self.pga_warning('max does not affect floating indv when mapped to ints')
        self.maxm = maxm

    def set_minm(self, minm):
        if self.mtype is not None and not isinstance(minm, self.mtype):
            raise TypeError('minm should be mtype')
        if self.mtype is int:
            self.pga_warning('min does not affect floating indv when mapped to ints')
        self.minm = minm

    def set_signbit(self, signbit):
        if not isinstance(signbit, bool):
            raise TypeError('signbit parameter should be a bool')
        if self.mtype is not None and self.mtype is float:
            self.pga_warning('Signbit is irrelevant when mapped to float')
        self.signbit = signbit

    @classmethod
    def gen_gene_map(cls, n_genes):
        gene_tag_len = math.ceil(math.log2(n_genes+1))
        # Gets a numpy array for each tag

        possible = [np.array([int(c) for c in \
                            format(n, "b").rjust(gene_tag_len, '0')]) \
                                        for n in range(2**gene_tag_len)]
        selected = random.sample(possible, n_genes)
        random.shuffle(selected)

        cls.gene_map = {n:s for n,s in enumerate(selected)}
        # Get gene matrix
        cls.gene_mat = \
            np.array([np.concatenate(cls.start_seq, cls.gene_map[key])] \
                                        for key in sorted(cls.gene_map.keys()))

    @classmethod
    def set_gene_map(cls, gene_map):
        cls.gene_map = gene_map

    @classmethod
    def set_dflt_val(cls, val):
        cls.dflt_val = val

    @classmethod
    def set_start_seq(cls, val):
        cls.start_seq = val

    def set_n_genes(self, n_genes):
        if not isinstance(n_genes, int):
            raise TypeError('n_genes should be int')
        if n_genes < 1:
            raise ValueError('n_genes should be greater than 0')
        self.n_genes = n_genes

        if self.gene_map is None or len(self.gene_map) != self.n_genes:
            self.gen_gene_map(n_genes)

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

        if 'dflt_val' in kargs:
            if kargs.get('dflt_val') != self.dflt_val:
                self.set_dflt_val(kargs.get('dflt_val'))
        elif 'dflt_val' in self.config:
            if self.config.get('dflt_val') != self.dflt_val:
                self.set_dflt_val(self.config.get('dflt_val'))
        else:
            self.config.get('dflt_val', self.dflt_val)

        if 'start_seq' in kargs:
            if kargs.get('start_seq') != self.start_seq:
                self.set_start_seq(kargs.get('start_seq'))
        elif 'start_seq' in self.config:
            if self.config.get('start_seq') != self.start_seq:
                self.set_start_seq(self.config.get('start_seq'))
        else:
            self.config.get('start_seq', self.start_seq)

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

    def map(self, vals, gene_mat, gene_len, is_int, use_signbit, minm, maxm):
        # Extract genes
        ext_genes = list(self.chromo.float_search(gene_mat))
        # Seperate into list and genes
        gene_nums = [x[0] for x in ext_genes]
        gene_arrs = np.array([vals[x[1]:x[2]] for x in ext_genes])
        # Convert the arrays into either integers or floats
        if self.mtype is int:
            gvals = self.cnvrt_to_int(gene_arrs, self.use_signbit)
        elif self.mtype is float:
            gvals = self.cnvrt_to_flt(gene_arrs, self.minm, self.maxm)

        val_dct = {}
        for n, v in zip(gene_nums, gene_arrs):
            val_dct.setdefault(n, []).append(v)
        if self.mtype is int:
            val_dct = {n:round(mean(item)) for n, item in val_dct.items()}
        elif self.mtype is float:
            val_dct = {n:mean(lst) for n, lst in val_dct.items()}
        # Replace blank values with 0
        return np.array([val_dct.get(n, self.dflt_val) for n in range(self.n_genes)])


    ''' Mapping function '''
    # Set up mapping methods, numba and nonnumba versions
    if 'numba' in sys.modules:

        @staticmethod
        @nb.jit(nopython=True)
        def cnvrt_to_int(arrs, use_signbit):
            # Create results array
            res = np.zeros(len(arrs))
            nbits = len(arrs[0])
            # Get what we multiply by
            if use_signbit:
                # Generate list of numbers
                mulby = np.append(np.zeros(1),\
                                  np.power(np.ones(nbits-1)*2, \
                                           np.arange(0, nbits-1)))
                for indx in range(len(arrs)):
                    res[indx] = np.sum(arrs[indx] * mulby)
                    if arrs[indx][0] == 1:
                        res[indx] = res[indx] * -1
            else:
                mulby = np.power(np.ones(nbits)*2, np.arange(0, nbits))
                for indx in range(len(arrs)):
                    res[indx] = np.sum(arrs[indx] * mulby)

            return res

        @staticmethod
        @nb.jit(nopython=True)
        def cnvrt_to_flt(arrs, minm, maxm):
            res = np.zeros(len(arrs))
            nbits = len(arrs[0])
            mulby = np.power(np.ones(nbits)*2, np.arange(0, nbits))
            oldmax = sum(mulby)
            for arr_indx in range(len(arrs)):
                res[gene_num] = sum(arrs[arr_indx] * mulby)
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
