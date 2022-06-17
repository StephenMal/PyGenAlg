from .basics.basicPopulation import basicPopulation

class fixedPopulation(basicPopulation):

    __slots__ = ()

    def __init__(self, **kargs):

        super().__init__(**kargs)
        self.varsize, self.minsize, self.maxsize = False, None, None


        if self.__class__ == fixedPopulation:
            self.set_params(**kargs)


    def raiseFixedSizeError(self, *args, **kargs):
        raise ValueError('Function call would alter size of fixed population')
    append = raiseFixedSizeError
    extend = raiseFixedSizeError
    insert = raiseFixedSizeError
    pop = raiseFixedSizeError

    def get_maxsize(self):
        return self.popsize
    def get_minsize(self):
        return self.popsize
    def get_varsize(self):
        return False

    def set_params(self, **kargs):

        for argname in ('varsize', 'maxsize', 'minsize'):
            if argname in kargs:
                kargs.pop(argname)
                self.pga_warning(f'Cannot set {argname} in fixedPopulation')

        # Some default values
        if self.popsize is None and 'popsize' not in kargs:
            self.set_popsize(self.config.get('popsize',200,dtype=int,mineq=2))

        super().set_params(**kargs)
