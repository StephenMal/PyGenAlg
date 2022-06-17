from ...basics import basicComponent
from .basicChromosome import basicChromosome
from ...exceptions import *
import numpy as np

try:
    import numba as nb
except:
    pass

class basicIndividual(basicComponent):

    __slots__ = ('chromo', 'fit', 'id', 'attrs')

    chromo_class = basicChromosome

    prev_id = 0

    def __init__(self, **kargs):

        super().__init__(**kargs)

        self.chromo, self.fit, self.id, self.attrs = None, None, None, None

        if self.__class__ == basicIndividual:
            self.set_params(**kargs)


    def generate(self, **kargs):
        self.fit = None
        if self.attrs is not None:
            del self.attrs

        chr = self.get_chromo(make_copy=False)
        if chr is None:
            chr = self.chromo_class(config=self.config, \
                                              log=self.log, \
                                              runsafe=self.runsafe)
        chr.generate(**kargs)

    def set_chromo(self, chromo):
        if not isinstance(chromo, self.chromo_class):
            if isinstance(chromo, dict):
                self.chromo = self.unpack_component(chromo)
            elif isinstance(chromo, np.ndarray):
                if self.chromo is None:
                    self.chromo = self.chromo_class(config=self.config,\
                                                    log=self.log,\
                                                    runsafe=self.runsafe)
                self.chromo.set_chromo(chromo)
            else:
                raise TypeError(f'Incorrect chromosome type ({type(chromo)})')
        else:
            self.chromo = chromo

    def set_id(self, new_id):
        if not isinstance(new_id, (str, int)):
            raise TypeError('Expected str or int')
        self.id = new_id

    @classmethod
    def reset_ids(cls):
        cls.prev_id = 0

    @classmethod
    def get_next_id(cls):
        cls.prev_id = cls.prev_id + 1
        return cls.prev_id


    def set_params(self, **kargs):

        if 'chromo' in kargs:
            self.set_chromo(kargs.get('chromo'))
        else:
            self.set_chromo(self.chromo_class(**kargs))

        if 'fit' in kargs:
            self.set_fit(kargs.get('fit'))
        elif self.fit is None and 'fit' in self.config:
            raise Exception('fit cannot be in config')

        if 'id' in kargs:
            self.set_id(kargs.get('id'))
        elif self.id is None:
            if 'id' in self.config:
                raise Exception('id cannot be in config')
            self.set_id(self.get_next_id())

        if 'attrs' in kargs:
            self.set_attrs(kargs.get('attrs'))
        elif self.attrs is None and 'attrs' in self.config:
            raise Exception('attrs cannot be in config')

        super().set_params(**kargs)

    def __hash__(self):
        return self.chromo.__hash__()

    def __eq__(self, other):
        return self.chromo.__eq__(other)

    def __del__(self):
        if self.chromo is not None:
            del self.chromo
        if self.attrs is not None:
            del self.attrs
        self.id, self.fit = None, None
        super().__del__()

    def to_list(self):
        if self.chromo is None:
            return []
        return self.chromo.to_list()

    def to_numpy(self, make_copy=None):
        if self.chromo is None:
            return np.array([])
        return self.chromo.to_numpy(make_copy=make_copy)

    def get_id(self):
        if self.id is None:
            raise MissingValue('id is not defined')
        return self.id

    def get_chromo(self, make_copy=False):
        if self.chromo is None:
            raise MissingValue('chromo is not defined')
        if make_copy:
            return copy(self.chromo)
        return self.chromo

    def get_dtype(self):
        return self.vals.get_dtype()

    def get_max(self):
        return self.chromo.get_maxv()

    def get_min(self):
        return self.chromo.get_minv()

    def __len__(self):
        if self.chromo is None:
            return 0
        return self.get_mapped().__len__()

    def __iter__(self):
        if self.chromo is None:
            return iter([])
        return self.get_mapped().__iter__()


    ''' Mapping '''

    @staticmethod
    def map(arr):
        raise NotImplementedError()

    def get_mapped(self):
        if self.chromo is None:
            raise MissingValue('Chromo must not be None to get mapped chromo')
        return self.map(self.chromo.to_numpy(make_copy=False))

    ''' attrs '''

    def clear_attrs(self):
        del self.attrs
        self.attrs = None

    def set_attr(self, attr_name, value):
        if self.runsafe:
            if not isinstance(attr_name, str):
                raise TypeError('attr_name should be str')
            if '.' in attr_name:
                raise ValueError('attr_name should not have .')
        if self.attrs is None:
            self.attrs = {}
        self.attrs.__setitem__(attr_name, value)

    def set_attrs(self, attrs):
        if self.runsafe:
            if any((not isinstance(key, str) for key in attrs.keys())):
                raise TypeError('Keys must be strings')
            if any(('.' in key for key in attrs.keys())):
                raise ValueError('Keys must not have .')
        self.attrs = attrs.copy()

    def get_attr(self, attr_name):
        if self.attrs is None:
            raise KeyError(f'{attr_name} is not a valid key')
        return self.attrs.__getitem__(attr_name)

    def get_attrs(self, make_copy=True):
        if self.attrs is None:
            return {}
        if make_copy:
            return self.attrs.copy()
        return self.attrs

    def update_attrs(self, dct):
        if not isinstance(dct, dict):
            raise TypeError('Expected dict')
        if self.runsafe:
            if any(('.' in key for key in dct.keys())):
                raise ValueError('Cannot contain . in keys')
            if not all((isinstance(key, str) for key in dct.keys())):
                raise TypeError('Keys must be str')
        if self.attrs is None:
            self.attrs = dct
        else:
            self.attrs.update(dct)

    def incr_attr(self, *args):
        if self.attrs is None:
            self.attrs = {}
        if len(args) == 2:
            if self.runsafe and not isinstance(args[0], str) and \
                                        not isinstance(args[1], (int, float)):
                raise TypeError('Expected str and int/float for incr_attr')
            self.attrs.__setitem__(args[0], self.attrs.setdefault(args[0], 0)+args[1])
        elif len(args) == 1:
            if self.runsafe and not isinstance(args[0], str):
                raise TypeError('Expected str for incr_attr')
            self.__setitem__(args[0], self.attrs.setdefault(args[0], 0)+1)

    def del_attr(self, attr_name):
        if self.attrs is None:
            return
        del self.attrs[attr_name]

    ''' Comparison '''

    try:
        @staticmethod
        @nb.jit(nopython=True, parallel=True)
        def _same(arr, other):
            if len(arr) != len(other):
                return False
            for i in nb.prange(len(arr)):
                if arr[i] != other[i]:
                    return False
            return True

        def same(self, other):
            if not isinstance(other, (basicIndividual, np.ndarray)):
                raise TypeError('Expected something inheriting from basicIndividual')
            if self.__len__() != len(other):
                return False
            if isinstance(other, basicIndividual):
                other = other.to_numpy(make_copy=False)
            return self._same(self.to_numpy(make_copy=False), other)
    except:
        def same(self, other):
            if not isinstance(other, (basicIndividual, np.ndarray)):
                raise TypeError('Expected something inheriting from basicIndividual')
            if len(arr) != len(other):
                return False
            if isinstance(other, basicIndividual):
                other = other.to_numpy(make_copy=False)
            for x, y in zip(arr, other):
                if x != y:
                    return False
            return True



    ''' Fitness '''

    def get_fit(self):
        if self.fit is None:
            raise MissingValue('Missing fit')
        return self.fit

    def set_fit(self, newfit):
        if isinstance(newfit, (int, np.integer, np.float_, str)):
            self.fit = float(newfit)
        elif isinstance(newfit, float):
            self.fit = newfit
        else:
            raise TypeError(f'fitness need to be int / float not {type(newfit)}')

    def __int__(self):
        if self.fit is None:
            raise MissingValue('Missing fit')
        return int(self.fit)

    def __float__(self):
        if self.fit is None:
            raise MissingValue('Missing fit')
        return float(self.fit)

    def __abs__(self):
        if self.fit is None:
            raise MissingValue('Missing fit')
        return abs(self.fit)

    ''' Export '''

    def __str__(self):
        return f'ID {self.get_id()}\tfit:{self.get_fit()}\n\t\t{self.to_list()}'

    def __repr__(self):
        return self.pack(compress=True).__str__()

    def to_dict(self):
        dct = super().to_dict()
        dct.update({'id':self.id,\
                    'fit':self.fit,\
                    'attrs':self.attrs,\
                    'chromo':self.chromo})
        return dct

    def pack(self, **kargs):
        dct = super().pack(**kargs)

        if self.id is not None:
            dct['id'] = self.id
        if self.fit is not None:
            dct['fit'] = self.fit
        if self.attrs is not None and len(self.attrs) > 0:
            dct['attrs'] = self.attrs.copy()
        if self.chromo is not None and len(self.chromo) > 0:
            dct['chromo'] = self.chromo.pack()
        return dct

    @classmethod
    def unpack(cls, dct):
        if 'chromo' in dct:
            dct['chromo'] = cls.unpack_component(dct['chromo'])
        if not isinstance(dct['chromo'], cls.chromo_class):
            raise TypeError(f'Expected chromosome of type {cls.chromo_class}')
        return super().unpack(dct)
