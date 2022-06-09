from .basics.basicEvaluator import basicEvaluator
from ..exceptions import *

try:
    import dill
except:
    pass

class customFunctionEvaluator(basicEvaluator):

    __slots__ ={'fxn':'The python function we are optimizing',\
                'inp_type':'How the individual should input itself to fxn'}

    DEFAULT_fxn = None
    DEFAULT_inp_type = 'numpy'

    def __init__(self, **kargs):

        super().__init__(**kargs)

        self.fxn, self.inp_type = None, self.DEFAULT_inp_type

        if self.__class__ == customFunctionEvaluator:
            self.set_params(**kargs)

    def set_fxn(self, fxn):
        if not callable(fxn):
            raise TypeError('Expected fxn to be callable')
        self.fxn = fxn

    def set_inp_type(self, inp_type):

        if not isinstance(inp_type, str):
            raise TypeError('Expected inp_type to be a str')
        inp_type = inp_type.lower()
        if not inp_type in ('list', 'numpy', 'indv', 'chromo'):
            raise ValueError('Expected list, numpy, indv, or chromo')

        self.inp_type = inp_type

    def evaluate(self, indv, **kargs):

        # Apply fitness function
        if self.inp_type == 'list':
            fit = self.fxn(indv.to_list())
            indv.set_fit(fit)
        elif self.inp_type == 'numpy':
            fit = self.fxn(indv.to_numpy(make_copy=False))
            indv.set_fit(fit)
        elif self.inp_type == 'indv':
            x = self.fxn(indv)
            # If returned a value, assume it is the new fit and place it
            if isinstance(x, (int, float)):
                indv.set_fit(x)
        elif self.inp_type == 'chromo':
            fit = self.fxn(indv.get_chromo(make_copy=False))
            indv.set_fit(fit)

        if kargs.get('update_minmax', True):
            self.set_max_indv(indv, fit=fit)
            self.set_min_indv(indv, fit=fit)

    def set_params(self, **kargs):

        if 'fxn' in kargs:
            self.set_fxn(kargs.get('fxn'))
        elif self.fxn is None and 'fxn' in self.config:
            self.set_fxn(self.config.get('fxn', callable=True))

        if 'inp_type' in kargs:
            self.set_inp_type(kargs.get('inp_type'))
        elif self.inp_type is None and ('inp_type' in self.config or \
                                            self.DEFAULT_inp_type is not None):
            self.set_inp_type(self.config.get('inp_type',self.DEFAULT_inp_type,\
                                options=('list','numpy','indv','chromo'),\
                                dtype=str))

        super().set_params(**kargs)



    def pack(self, compress=True, incl_bsc_cmp=False):
        raise NotImplementedError('Cannot pack customFunctionEvaluator')

    def unpack(self):
        raise NotImplementedError('Cannot unpack customFunctionEvaluator')
