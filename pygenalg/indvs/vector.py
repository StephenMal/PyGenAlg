from .basics import basicChromosome, basicIndividual
import sys

try:
    import numba as nb
except:
    pass
class vectorChromosome(basicChromosome):

    __slots__ = ()

    DEFAULT_length = None
    DEFAULT_maxv = None
    DEFAULT_minv = None
    DEFAULT_dtype = None
    DEFAULT_dsize = 64
    DEFAULT_np_dtype = None
    DEFAULT_minlen = None
    DEFAULT_maxlen = None
    DEFAULT_varlen = False

    def __init__(self, **kargs):
        super().__init__(**kargs)

        if self.__class__ == vectorChromosome:
            self.set_params(**kargs)
            self.varlen = False


    if 'numba' in sys.modules:

        @staticmethod
        @nb.jit(nopython=True)
        def _hash(arr):
            if len(arr) < 1000:
                return np.sum(\
                    np.logspace(1,len(arr),num=len(arr)-1,base=10,dtype='int')*arr)
            else:
                rng = np.random.RandomState(89)
                inds = rng.randint(low=0, high=a.size, size=1000)
                b = a.flat[inds]
                b.flags.writeable = False
                return hash(b.data)


        def __hash__(self):
            return self._hash(self.vals)

    else:

        def __hash__(self):
            if len(arr) < 1000:
                return np.sum(\
                    np.logspace(1,len(arr),num=len(arr)-1,base=10,dtype='int')*arr)
            else:
                rng = np.random.RandomState(89)
                inds = rng.randint(low=0, high=a.size, size=1000)
                b = a.flat[inds]
                b.flags.writeable = False
                return hash(b.data)




    def set_minlen(self, minlen):
        self.pga_warning('minlen must be None')
        self.minlen = None
    def set_maxlen(self, maxlen):
        self.pga_warning('maxlen must be None')
        self.maxlen = None
    def set_varlen(self, vlen):
        self.pga_warning('varlen must be None or False')
        self.varlen = self.DEFAULT_varlen

    def set_params(self, **kargs):

        for argname in ('minlen', 'maxlen', 'varlen'):
            try:
                kargs.pop(argname)
            except KeyError:
                pass
        super().set_params(**kargs)

    def pack(self, **kargs):
        return super().pack(**kargs)

class vectorIndividual(basicIndividual):

    __slots__ = ()

    chromo_class = vectorChromosome

    def __init__(self, **kargs):

        super().__init__(**kargs)

        if self.__class__ == vectorIndividual:
            self.set_params(**kargs)

    def map(self, arr):
        return arr.copy()
