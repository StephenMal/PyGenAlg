from ...basics import basicComponent
from ...exceptions import *
from collections import namedtuple

import numpy as np
import numpy.random as npr
import random as rand
import sys

try:
    import numba as nb
except:
    pass

class basicChromosome(basicComponent):

    __slots__ = ('vals', 'maxv', 'minv', 'dtype', 'dsize', 'np_dtype', \
                 'minlen', 'maxlen', 'varlen', 'length')

    DEFAULT_length = None
    DEFAULT_maxv = None
    DEFAULT_minv = None
    DEFAULT_dtype = None
    DEFAULT_dsize = 64
    DEFAULT_np_dtype = None
    DEFAULT_minlen = None
    DEFAULT_maxlen = None
    DEFAULT_varlen = None

    def __init__(self, **kargs):

        # Content
        self.vals, self.length = None, None

        # Value Parameters
        self.maxv,  self.minv = None, None

        # Dtype (python and numpy)
        self.dtype, self.dsize, self.np_dtype = None, None, None

        # Length Parameters
        self.minlen, self.maxlen, self.varlen = None, None, None


        # Initialize basic component
        super().__init__(**kargs)

        # Run set_params only if basicCHromosome is being made, not if its child
        #   is calling this method
        if self.__class__ == basicChromosome:
            self.set_params(**kargs)

    # Delete
    def __del__(self):
        if self.vals is not None:
            del self.vals
        super().__del__()

    def __len__(self):
        if self.vals is None:
            return 0
        return self.vals.__len__()

    def __str__(self):
        if self.vals is None:
            return 'Empty Chromosome'
        return self.vals.__str__()

    def __repr__(self):
        if self.vals is None:
            return 'Empty Chromosome'
        return self.vals.__repr__()

    def __iter__(self):
        if self.vals is None:
            self.log.warn('Attempting to iterate through empty chromosome')
            return iter([])
        return self.vals.tolist().__iter__()

    def __eq__(self, other):
        if isinstance(other, np.ndarray):
            return np.array_equal(self.vals, other)
        elif isinstance(other, (basicChromosome, basicIndividual)):
            return np.array_equal(self.vals, other.to_numpy(make_copy=False))

    def determine_npdtype(self):
        if self.dtype is not None:
            if self.dsize is None:
                self.pga_warning('dsize not provided.  Assumes 64 when not provided')
            if self.dtype is int:
                if self.minv is None or self.minv < 0:
                    if self.dsize is None:
                        self.np_dtype = np.int64
                    elif self.dsize == 8:
                        self.np_dtype = np.int8
                    elif self.dsize == 16:
                        self.np_dtype = np.int16
                    elif self.dsize == 32:
                        self.np_dtype = np.int32
                    elif self.dsize == 64:
                        self.np_dtype = np.int64
                    else:
                        raise ValueError('dtype should be 8, 16, 32, or 64 for int')
                elif self.minv is None or self.minv >= 0:
                    if self.dsize is None:
                        self.np_dtype = np.uint64
                    elif self.dsize == 8:
                        self.np_dtype = np.uint8
                    elif self.dsize == 16:
                        self.np_dtype = np.uint16
                    elif self.dsize == 32:
                        self.np_dtype = np.uint32
                    elif self.dsize == 64:
                        self.np_dtype = np.uint64
                    else:
                        raise ValueError('dtype should be 8, 16, 32, or 64 for uint')
            elif self.dtype is float:
                if self.dsize is None:
                    self.np_dtype = np.float64
                elif self.dsize == 8:
                    raise ValueError('Cannot have dsize of 8 with float')
                elif self.dsize == 16:
                    self.np_dtype = np.float16
                elif self.dsize == 32:
                    self.np_dtype = np.float32
                elif self.dsize == 64:
                    self.np_dtype = np.float64
                else:
                    raise ValueError('dsize should be 16, 32, or 64 for float')
            else:
                raise ValueError('dtype should be int or float')

    # TO DO move parameter setting / getting into seperate functions
    #   to allow more OOP approach

    def get_dtype(self):
        return self.dtype
    def get_dsize(self):
        return self.dsize
    def get_varlen(self):
        return self.varlen
    def get_length(self):
        return self.length

    def set_dtype(self, dtype):

        if isinstance(dtype, str):
            dtype_lower = dtype.lower()
            if dtype_lower == 'float':
                dtype = float
            elif dtype_lower == 'int' or dtype_lower == 'integer':
                dtype = int

        if not isinstance(dtype, type):
            raise TypeError('dtype should be a type')
        elif dtype is not float and dtype is not int:
            raise TypeError('dtype should be float or int')

        self.dtype = dtype

    def set_dsize(self, dsize):
        if not isinstance(dsize, int):
            raise TypeError('Expected int for dsize')
        if dsize not in (8, 16, 32, 64):
            raise ValueError('Expected 8, 16, 32, or 64 for dsize')

        self.dsize = dsize

    def set_minv(self, minv):
        if self.dtype is not None:
            if not isinstance(minv, self.dtype):
                raise TypeError('Min does not match dtype')
        elif not isinstance(minv, (float, int)):
            raise TypeError('Expected an int or a float for min')
        self.minv = minv

    def set_maxv(self, maxv):
        if self.dtype is not None:
            if not isinstance(maxv, self.dtype):
                raise TypeError('Max does not match dtype')
        elif not isinstance(maxv, (float, int)):
            raise TypeError('Expected an int or a float for max')
        self.maxv = maxv

    def set_minlen(self, minlen):
        # minlen
        if not isinstance(minlen, int):
            raise TypeError('Expected an int value for minlen')
        if minlen <= 0:
            raise ValueError('minlen should be greater than 0')
        self.minlen = minlen

    def set_maxlen(self, maxlen):
        if not isinstance(maxlen, int):
            raise TypeError('Expected an int value for maxlen')
        if maxlen <= 0:
            raise ValueError('maxlen should be greater than 0')
        self.maxlen = maxlen

    def set_varlen(self, vlen):
        if not isinstance(vlen, bool):
            raise TypeError('Expected a boolean')

        self.varlen = vlen

    def set_length(self, length):
        if not isinstance(length, int) and isinstance(length, float) and \
                            length.is_integer():
            length = int(length)
        if not isinstance(length, int):
            raise TypeError('Expected an int for length')

        if self.minlen is not None and length < self.minlen:
            raise ValueError(\
                f'length {length} is less than min {self.minlen}')

        if self.maxlen is not None and length > self.maxlen:
            raise ValueError(\
                f'length {length} is more than max {self.maxlen}')

        self.length = length

    def set_chromo(self, vals, validate=None):
        if isinstance(vals, list):
            if self.dtype is None:
                dtype_set = set([type(x) for x in vals])
                if len(dtype_set) == 1:
                    if float in dtype_set:
                        self.dtype = float
                    elif int in dtype_set:
                        self.dtype = int
                    else:
                        raise TypeError('dtype should be float / int')
                elif len(dtype_set) == 2:
                    if float in dtype_set and int in dtype_set:
                        self.pga_warning('dtype was not provided and set_chromo'+\
                                         ' was given a list with multiple dtypes.'+\
                                         ' In this instance we assume float')
                        self.dtype = float
                        vals = [float(val) for val in vals]
                # Attempt to guess npdtype
                self.determine_npdtype()

            if self.np_dtype is None:
                raise RuntimeError('np_dtype cannot be None')
            vals = np.array(vals, dtype=self.np_dtype)
            if validate is True or (self.runsafe and validate is None):
                self.validate(vals)
            self.vals = vals
        elif isinstance(vals, np.ndarray):
            if validate is True or (self.runsafe and validate is None):
                self.validate(vals)
            self.vals = vals.copy()
        else:
            raise TypeError('vals must be list or a numpy array')

    # Set the parameters
    def set_params(self, **kargs):
        # Dtype verification
        if 'dtype' in kargs:
            self.set_dtype(kargs.get('dtype'))
        elif self.dtype is None and ('dtype' in self.config or \
                                        self.DEFAULT_dtype is not None):
            self.set_dtype(self.config.get('dtype', self.DEFAULT_dtype, \
                                                        options=(int, float)))

        # Dsize verification
        if 'dsize' in kargs:
            self.set_dsize(kargs.get('dsize'))
        elif self.dsize is None and ('dsize' in self.config or
                                        self.DEFAULT_dsize is not None):
            self.set_dsize(self.config.get('dsize', self.DEFAULT_dsize,\
                                            options=(8,16,32,64), dtype=int))

        self.determine_npdtype()

        # Maximum
        if 'max' in kargs:
            self.set_maxv(kargs.get('max'))
        elif self.maxv is None and 'max' in (self.config or \
                                                self.DEFAULT_maxv is not None):
            self.set_maxv(self.config.get('max', self.DEFAULT_maxv, \
                                                            dtype=(float,int)))

        # Minimum
        if 'min' in kargs:
            self.set_minv(kargs.get('min'))
        elif self.minv is None and 'min' in (self.config or \
                                                self.DEFAULT_minv is not None):
            self.set_minv(self.config.get('min', self.DEFAULT_minv,\
                                                        dtype=(float,int)))


        if self.maxv is not None and self.minv is not None:
            if self.maxv <= self.minv:
                raise ValueError('max needs to be greater than min')
            if self.minv >= self.maxv:
                raise ValueError('min needs to be less than max')

        # minlen
        if 'minlen' in kargs:
            self.set_minlen(kargs.get('minlen'))
        elif self.minlen is None and 'minlen' in self.config:
            self.set_minlen(self.config.get('minlen', dtype=int, mineq=1))

        # maxlen
        if 'maxlen' in kargs:
            self.set_maxlen(kargs.get('maxlen'))
        elif self.maxlen is None and 'maxlen' in self.config:
            self.set_maxlen(self.config.get('maxlen', dtype=int, mineq=1))

        # Handle varlen
        if 'varlen' in kargs:
            self.set_varlen(kargs.get('varlen'))

        if self.varlen is True:
            if self.maxlen is not None and self.minlen is not None and \
                                                self.maxlen == self.minlen:
                raise ValueError('maxlen must not equal minlen when '+\
                                 'varlen is True')
        elif self.varlen is False:
            if self.maxlen is not None and self.minlen is not None and \
                                                self.maxlen != self.minlen:
                raise ValueError('maxlen must equal minlen when '+\
                                 'varlen is True')

        # Length
        if 'length' in kargs:
            self.set_length(kargs.get('length'))
        elif self.length is None and 'length' in self.config:
            self.set_length(self.config.get('length', dtype=int, mineq=1))

        # Verify min / max lengths work
        if self.maxlen is not None and self.maxlen <= 0:
            raise ValueError('maxlen must be greater than 0')
        if self.minlen is not None and self.minlen <= 0:
            raise ValueError('minlen must be greater than 0')
        if self.maxlen is not None and self.minlen is not None and \
                                            self.maxlen < self.minlen:
            raise ValueError('Cannot maxlen is less than minlen')

        if 'vals' in kargs:
            self.set_chromo(kargs.get('vals'), validate=kargs.get('validate'))

    # Generate
    def generate(self, **kargs):
        # If provided extra kargs, set them
        if len(kargs) > 0:
            self.set_params(**kargs)

        if self.varlen is True:
            if self.minlen is None or self.maxlen is None:
                raise MissingValue('minlen and/or maxlen is missing')
            self.length = rand.randint(self.minlen, self.maxlen+1)
        elif self.length is None:
            if self.maxlen is not None and self.minlen is not None:
                if self.maxlen == self.minlen:
                    self.length = self.maxlen
            elif self.maxlen is not None:
                self.length = self.maxlen
            elif self.minlen is not None:
                self.length = self.minlen
            else:
                raise MissingValue('If fixed length, need length provided')

        for arg in ('minv', 'maxv', 'length', 'dtype'):
            if getattr(self, arg) is None:
                raise MissingValue(f'Missing {arg} which is necessary to generate values')

        self.determine_npdtype()

        # Create value depending on datatype
        if self.dtype is int:
            vals = self.nprng.integers(self.minv, self.maxv, size=self.length,\
                                       dtype=self.np_dtype, endpoint=True)
        elif self.dtype is float:
            vals = self.nprng.uniform(self.minv, self.maxv, size=self.length)
        else:
            raise ValueError('dtype should be int or float')

        self.set_chromo(vals)

    if 'numba' in sys.modules:
        @staticmethod
        @nb.jit(nopython=True)
        def _get_val_stats(vals):
            return vals.min(), vals.max(), vals.size, vals.dtype
    else:
        @staticmethod
        def _get_val_stats(vals):
            return vals.min(), vals.max(), vals.size, vals.dtype

    # Validate the chromosome
    def validate(self, vals):
        # Get stats
        minv, maxv, length, dtype = self._get_val_stats(vals)

        if self.length is not None and self.length != len(vals):
            raise ValueError('length does not match listed length')
        if self.varlen is True or (self.varlen is None and \
                                   self.minlen is not None and \
                                   self.maxlen is not None):
            if self.minlen is None or self.maxlen is None:
                raise MissingValue('Need minlength and maxlength')
            if self.length < self.minlen or self.length > self.maxlen:
                raise ValueError('Length is out of bounds')
        elif self.varlen is False:
            if self.length is None:
                if self.minlen is not None and self.maxlen is not None:
                    if self.minlen != self.maxlen:
                        raise ValueError('minlength != maxlength')
                    if len(vals) != self.minlen:
                        raise ValueError('length != min/max length')
                elif self.minlen is not None:
                    if len(vals) != self.minlen:
                        raise ValueError('length != min/max length')
                elif self.maxlen is not None:
                    if len(vals) != self.maxlen:
                        raise ValueError('length != min/max length')
                else:
                    self.pga_warning('Did not provide length for validation,'+\
                                     'saving this length as future length')

                    self.length = len(vals)
            if self.minlen is not None and self.length < self.minlen:
                raise ValueError('Length is out of bounds')
            if self.maxlen is not None and self.length > self.maxlen:
                raise ValueError('Length is out of bounds')

        if self.np_dtype is None:
            raise MissingValue('missing np_dtype')
        elif self.dtype is int:
            if not np.issubdtype(dtype, self.np_dtype):
                raise TypeError('vals arr is not of int dtype')
        elif self.dtype is float:
            if not np.issubdtype(dtype, self.np_dtype):
                raise TypeError('vals arr is not of float dtype')
        elif self.dtype is None:
            raise MissingValue('dtype needs to be specified')
        else:
            raise ValueError('invalid dtype')

        if self.max is None or self.min is None:
            raise MissingValue('max and min must not be None')
        if maxv > max(vals) or minv < min(vals):
            raise ValueError('out of value range')

    # Get from index
    def __getitem__(self, indx):
        if self.dtype is int:
            return int(self.vals.__getitem__(indx))
        elif self.dtype is float:
            return float(self.vals.__getitem__(indx))
        elif self.dtype is None:
            raise MissingValue('Need dtype')
        else:
            raise ValueError('Incorrect dtype')

    def get(self, indx):
        return self.__getitem__(indx)

    def getnp(self, indx):
        return self.vals.__getitem__(indx)

    # Set the item
    def __setitem__(self, indx, item):
        if not isinstance(item, self.dtype):
            item = self._turn_into_correct_dtype(item)
        self.vals.__setitem__(indx, item)

    def _turn_into_correct_dtype(self, item):

        if self.dtype is int:
            if isinstance(item, float):
                if item.is_integer():
                    return int(item)
                else:
                    self.pga_warning('Placing float in int chromosome will lead '+\
                              ' to loss in precision')
                    return int(round(item))
            else:
                try:
                    return int(item)
                except:
                    raise TypeError('Failed to convert value to int')

        elif self.dtype is float:
            try:
                return float(item)
            except:
                raise TypeError('Failed to convert value to float')
        elif self.dtype is None:
            raise MissingValue('dtype must be specified')
        else:
            raise ValueError('Dtype must be int or float')

    ''' Editing Chromosome '''

    # Append
    def append(self, item):
        if self.varlen is True:
            if not isinstance(item, self.dtype):
                item = self._turn_into_correct_dtype(item)
            self.set_chromo(np.append(self.vals, item))
        else:
            raise LengthError('Cannot append to fixed length chromo')

    # Extend (expands with iterable)
    def extend(self, item):
        if self.varlen is True:
            if not isinstance(item, self.dtype):
                item = self._turn_into_correct_dtype(item)
            self.set_chromo(np.append(self.vals, item))
        else:
            raise LengthError('Cannot extend to fixed length chromo')

    def insert(self, indx, item):
        if self.varlen is True:
            if not isinstance(item, self.dtype):
                item = self._turn_into_correct_dtype(item)
            self.set_chromo(np.insert(self.vals, indx, item))
        else:
            raise LengthError('Cannot insert to fixed length chromo')

    # Replace
    def replace(self, old, new):
        if not isinstance(new, self.dtype):
            new = self._turn_into_correct_dtype(new)
        self.set_chromo(np.where(self.vals==old, new, self.vals))

    def pop(self, indx):
        ret = self.__getitem__(indx)
        self.set_chromo(np.delete(self.vals, indx))
        return ret

    def remove(self, item):
        if not isinstance(item, self.dtype):
            item = self._turn_into_correct_dtype(item)
        self.set_chromo(np.delete(self.vals, self.vals==item))

    ''' SEARCHING '''

    if 'numba' in sys.modules:
        # Utilizes numba's jit to quickly perform a search of the array
        @staticmethod
        @nb.jit(nopython=True)
        def _search_npsubarr(arr, subarr):
            # Get lengths
            arrlen = len(arr)
            sublen = len(subarr)
            # Return empty if sublen is longer than arrlen
            if sublen > arrlen:
                return

            match_found = True
            for i in range(0, sublen):
                if arr[i] != subarr[i]:
                    match_found = False
            if match_found == True:
                yield 0

            subsum = 0
            for i in range(sublen):
                subsum += subarr[i]

            cursum = 0
            for i in range(sublen):
                cursum += arr[i]

            for i in range(1, arrlen - sublen + 1):
                cursum += arr[i + sublen - 1] - arr[i - 1]
                if cursum == subsum:
                    match_found = True
                    for j in range(sublen):
                        if arr[i+j] != subarr[j]:
                            match_found = False
                            break
                    if match_found == True:
                        yield i
        @staticmethod
        @nb.jit(nopython=True)
        def _npflt_is_int(val):
            if val == round(val):
                return True
            else:
                return False
    else:
        # Rabin-Karp Algorithm, more efficient with python interpreter than
        #   checking each possible subarr
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

            # Get hash of subarr
            subhash = 0
            for x in subarr:
                subhash += hash(x)

            # Get hash of beginning
            curhash = 0
            for x in arr[:sublen]:
                curhash += hash(x)

            for i in range(1, arrlen - sublen + 1):
                curhash += hash(arr[i + sublen - 1]) - hash(arr[i - 1])
                if subhash == curhash:
                    match_found = True
                    for j in range(sublen):
                        if arr[i + j] != subarr[j]:
                            match_found = False
                    if match_found == True:
                        yield i

        @staticmethod
        def _npflt_is_int(val):
            if val == round(val):
                return True
            else:
                return False

    # Returns a generater that will iterate through the indicies of where the
    #    values or the subarrays start
    def search(self, item, as_list=False):
        # If singular item just find all the locations
        if isinstance(item, (int, float, np.integer, np.float_)):
            if self.dtype is int and \
                    (isinstance(item, float) and item.is_integer()) or \
                    (isinstance(item, np.float_) and self._npflt_is_int(item)):
                if as_list:
                    return []
                return iter([])
            if as_list:
                return np.where(self.vals == item)[0].tolist()
            return iter(np.where(self.vals == item)[0].tolist())
        elif isinstance(item, (list, np.ndarray)):
            # If a list, convert to numpy array
            if isinstance(item, list) and not isinstance(item, np.ndarray):
                if self.dtype is int and \
                    not all([isinstance(i, (int, np.integer)) for i in item]):
                    raise TypeError('Incorrect dtype in list')
                item = np.array(item, dtype=self.np_dtype)
            # Check for the numpy subarr
            if isinstance(item, np.ndarray):
                if as_list:
                    return list(self._search_npsubarr(self.vals, item))
                return self._search_npsubarr(self.vals, item)
        else:
            raise TypeError('Expected int, float, list, or np ndarray')

    def __contains__(self, item):
        if isinstance(item, (int, float)):
            return self.vals.__contains__(item)
        if isinstance(item, list):
            item = np.array(list)
        if isinstance(item, np.ndarray):
            return self._contains_subarr(self.vals, item)
        raise TypeError('Expected int, float, or list')


    ''' Counting '''

    # Stats
    def count(self, item=None):
        if item is not None:
            return int(np.count_nonzero(self.vals == 2))
        if self.dtype is int:
            return {int(key):int(item) for key,item in \
                                np.unique(self.vals, return_counts=True)}
        elif self.dtype is float:
            return {float(key):int(item) for key,item in \
                                np.unique(self.vals, return_counts=True)}
        elif self.dtype is None:
            raise MissingValue('Need dtype')
        else:
            raise ValueError('Expected float or int for dtype')

    def returnCounter(self):
        return Counter(self.vals)

    def indicies(self, *args, **kargs):
        return np.indices(self.vals, *args, **kargs)

    def where(self, *args, **kargs):
        return np.where(self.vals, *args, **kargs)

    ''' Reordering '''

    def reverse(self, item):
        self.set_chromo(np.flip(self.vals))

    def sort(self, **kargs):
        self.set_chromo(np.sort(self.vals, **kargs))

    if 'numbda' in sys.modules:
        @staticmethod
        @nb.jit(nopython=True)
        def _min(vals, minval, maxval):
            minimum = maxval
            for indx in nb.prange(len(vals)):
                if vals[indx] == minval:
                    return minval
                if vals[indx] < minimum:
                    minimum = x
            return minimum
        @staticmethod
        @nb.jit(nopython=True)
        def _min(vals, minval, maxval):
            maximum = minval
            for indx in nb.prange(len(vals)):
                if vals[indx] == maxval:
                    return maxval
                if vals[indx] > maximum:
                    maximum = x
            return maximum
        # Getting the minimum
        def min(self):
            if self.dtype is float:
                return float(self._min(self.vals, self.minv, self.maxv))
            elif self.dtype is int:
                return int(self._min(self.vals, self.minv, self.maxv))
            elif self.dtype is None:
                raise MissingValue('Need dtype')
            else:
                raise ValueError('Expected int or float')
        # Getting the maximum
        def max(self):
            if self.dtype is float:
                return float(self._max(self.vals))
            elif self.dtype is int:
                return int(self._max(self.vals))
            elif self.dtype is None:
                raise MissingValue('Need dtype')
            else:
                raise ValueError('Expected int or float')
    else:
        # Getting the minimum
        def min(self):
            if self.dtype is float:
                return float(np.amin(self.vals))
            elif self.dtype is int:
                return int(np.amin(self.vals))
            elif self.dtype is None:
                raise MissingValue('Need dtype')
            else:
                raise ValueError('Expected int or float')
        # Getting the maximum
        def max(self):
            if self.dtype is float:
                return float(np.amax(self.vals))
            elif self.dtype is int:
                return int(np.amax(self.vals))
            elif self.dtype is None:
                raise MissingValue('Need dtype')
            else:
                raise ValueError('Expected int or float')

    def get_minv(self):
        return self.minv

    def get_maxv(self):
        return self.maxv

    # Splits array into subarrays
    def split(self, n, allow_uneven_split=False):
        try:
            return np.split(self.vals, n)
        except ValueError as e:
            if allow_uneven_split:
                return np.array_split(self.vals, n)
            raise e

    ''' Exporting '''

    # Exporting to list
    def to_list(self):
        if self.vals is None or len(self.vals) == 0:
            return []
        return self.vals.tolist()

    # Returning copy of vals
    def to_numpy(self, make_copy=True):
        if make_copy is True or make_copy is None:
            return np.copy(self.vals)
        return self.vals

    def to_dict(self):
        dct = super().to_dict()
        dct.update({'vals':self.vals.to_list(),\
                    'vrange':(self.minv, self.maxv),\
                    'dtype':self.dtype,\
                    'np_dtype':self.np_dtype,\
                    'lrange':(self.minlen, self.maxlen, self.varlen)})
        return dct

    __slots__ = ('vals', 'maxv', 'minv', 'dtype', 'dsize', 'np_dtype', \
                 'minlen', 'maxlen', 'varlen', 'length')

    @staticmethod
    def _compress_vals(vals):
        return vals

    @staticmethod
    def _decompress_vals(vals):
        return vals

    def pack(self, **kargs):

        dct = super().pack(**kargs)

        vals = self.vals
        if vals is None or len(vals) == 0:
            vals = []
        elif isinstance(vals, np.ndarray):
            vals = vals.tolist()

        dct['vals'] = vals

        if kargs.get('incl_defs', False):
            dct.update({'maxv':self.maxv, 'minv':self.minv, 'dtype':self.dtype.__name__,\
                        'dsize':self.dsize})
            if self.get_varlen() is True:
                dct.update({'minlen':self.minlen, 'maxlen':self.maxlen,\
                            'varlen':True})
            elif self.get_varlen() is False:
                dct.update({'varlen':False})
        else:
            if self.maxv != self.DEFAULT_maxv:
                dct['max'] = self.maxv
            if self.minv != self.DEFAULT_minv:
                dct['min'] = self.minv
            if self.dtype != self.DEFAULT_dtype:
                dct['dtype'] = self.dtype.__name__
            if self.dsize != self.DEFAULT_dsize:
                dct['dsize'] = self.dsize
            if self.get_varlen() != self.DEFAULT_varlen:
                dct['varlen'] = self.get_varlen()
            if self.get_varlen():
                dct.update({'minlen':self.minlen, 'maxlen':self.maxlen})

        return dct

    @classmethod
    def unpack(cls, dct):

        for argname in ('vals', ):
            if argname not in dct:
                raise MissingPackingVal(argname)

        if cls.DEFAULT_dtype is None and 'dtype' not in dct:
            raise MissingPackingVal('dtype')

        if 'dtype' in dct and isinstance(dct['dtype'], str):
            if dct['dtype'] == 'float':
                dct['dtype'] = float
            elif dct['dtype'] == 'int':
                dct['dtype'] = int

        if 'varlen' in dct:
            if dct['varlen'] == 0 or dct['varlen'] == False:
                dct['varlen'] = False
                if 'length' not in dct:
                    dct['length'] = len(dct['vals'])
            elif dct['varlen'] == 1 or dct['varlen'] == True:
                dct['varlen'] = True
                for argname in ('minlen', 'maxlen'):
                    if argname not in dct:
                        raise MissingPackingVal(argname)

        return super().unpack(dct)
