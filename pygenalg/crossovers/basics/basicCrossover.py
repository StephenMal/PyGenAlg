from ...basics import basicComponent
from ...exceptions import *

class basicCrossover(basicComponent):


    __slots__ = {'xovrate':\
                    'Float btwn 0.0 - 1.0 representing possibility of crossover',\
                 'n_parents':'Int number of parents needed for crossover',\
                 'varlen_okay':'Boolean of variable length chromosomes will work'}

    # DEFAULT VALUES #
    DEFAULT_xovrate = 0.9
    DEFAULT_n_parents = None
    DEFAULT_varlen_okay = False

    def __init__(self, **kargs):
        super().__init__(**kargs)

        # New variables
        self.xovrate, self.n_parents, self.varlen_okay = None, None, None

        if self.__class__ == basicCrossover:
            self.set_params(**kargs)

    def set_n_parents(self, n_parents):
        if not isinstance(n_parents, int):
            if isinstance(n_parents, float) and n_parents.is_integer():
                n_parents = int(n_parents)
            else:
                raise TypeError('Expected int for n_parents')
        self.n_parents = n_parents

    def set_xovrate(self, xovrate):
        if not isinstance(xovrate, float):
            if isinstance(xovrate, int):
                xovrate = float(xovrate)
            else:
                raise TypeError('Expected float for xovrate')
        if xovrate < 0 or xovrate > 1:
            raise ValueError('xovrate should be between 0 and 1')
        self.xovrate = xovrate

    def set_varlen_okay(self, varlen_okay):
        if not isinstance(varlen_okay, bool):
            if isinstance(varlen_okay, int):
                varlen_okay = self.int_to_bool(varlen_okay)
            else:
                raise TypeError('Expected bool')
        self.varlen_okay = varlen_okay

    def set_params(self, **kargs):
        if 'xovrate' in kargs:
            self.set_xovrate(kargs.get('xovrate'))
        elif self.xovrate is None and 'xovrate' in self.config:
            self.set_xovrate(self.config.get('xovrate', dtype=float, \
                                             mineq=0.0, maxeq=1.0))
        if 'xov_n_parents' in kargs:
            self.set_n_parents(kargs.get('xov_n_parents'))
        elif self.n_parents is None and ('xov_n_parents' in self.config or\
                                          self.DEFAULT_n_parents is not None):
            self.set_n_parents(self.config.get('xov_n_parents',\
                                    self.DEFAULT_n_parents, dtype=int, mineq=1))

        if 'xov_varlen_okay' in kargs:
            self.set_varlen(kargs.get('xov_varlen_okay'))
        elif self.varlen_okay is None and ('xov_varlen_okay' in self.config or\
                                        self.DEFAULT_varlen_okay is not None):
            self.set_varlen_okay(self.config.get('xov_varlen_okay', \
                                    self.DEFAULT_varlen_okay, dtype=bool))

        super().set_params(**kargs)

    def get_xov_rate(self):
        if self.xov_rate is None:
            raise MissingValue('Missing xov_rate')
        return self.xov_rate

    def get_n_parents_needed(self):
        if self.n_parents is None:
            raise MissingValue('Missing n_parents')
        return self.n_parents

    def get_varlen_okay(self):
        if self.varlen_okay is None:
            raise MissingValue('Missing varlen_okay')
        return self.varlen_okay

    @staticmethod
    def _cross(chromos):
        raise NotImplementedError

    # Creates a new set of individuals from the parents
    def cross_indvs(self, *items):

        if random.random() < self.xovrate:
            return self._cross(\
                            [indv.to_numpy(make_copy=False) for indv in items])
        else:
            if self.runsafe:
                if not all([isinstance(indv, (basicChromosome, basicIndividual)) \
                                                            for indv in items]):
                    raise TypeError('Expected a list of items')
                if len(grouping) != self.n_parents:
                    raise ValueError(f'Expected {self.n_parents} parents per group')
                if not self.varlen_okay:
                    if not all([len(indv)==length for indv in grouping]):
                        raise ValueError('Varlength is not acceptable')
            return [indv.to_numpy(make_copy=True) for indv in items]

    # Creates a new batch of children
    def cross_batch(self, groups):
        # If running safe, check debug
        if self.runsafe:
            # See if we can get the number needed
            try:
                num_needed = self.get_n_parents_needed()
            except MissingValue:
                raise MissingValue('Failed to get the number of parents needed')
            # Verify we were given a list or a tuple
            if not isinstance(groups, (list, tuple)):
                raise TypeError('Expected 2D list of individuals, not '+\
                                f'{type(groups)}')
            # Verify each item inside the list/tuple is a list/tuple and those
            #   contain basic Individuals and has the correct num needed
            if not all((isinstance(grp, (list, tuple)) and \
                all((isinstance(indv, (basicIndividual)) for indv in grp)) and \
                    len(grp) == num_needed
                        for grp in groups)):
                raise TypeError('Expected 2D list of individuals')

        # END of runsafe

        children = []
        for group in groups:
            ch = self._cross([indv.to_numpy(make_copy=False) for indv in group])
            children.extend(ch)
        return children

    def pack(self, **kargs):
        if self.xovrate is None:
            raise MissingPackingVal(argname='xovrate')
        if self.n_parents is None:
            raise MissingPackingVal(argname='n_parents')
        if self.varlen_okay is None:
            raise MissingPackingVal(argname='varlen_okay')


        dct = super().pack(**kargs)

        if kargs.get('incl_defs', False):
            dct.update({'xovrate':self.xovrate,\
                        'xov_n_parents':self.n_parents,\
                        'xov_varlen_okay':self.varlen_okay})
        else: # If not including defaults, double check
            if self.xovrate != self.DEFAULT_xovrate:
                dct['xovrate'] = self.xovrate
            if self.n_parents != self.DEFAULT_n_parents:
                dct['xov_n_parents'] = self.n_parents
            if self.varlen_okay != self.DEFAULT_n_parents:
                dct['xov_varlen_okay'] = self.varlen_okay

        return dct

    @classmethod
    def unpack(cls, dct):
        if cls.DEFAULT_xovrate is None and 'xovrate' not in dct:
            raise MissingPackingVal(argname='xovrate')
        if cls.DEFAULT_n_parents is None and 'xov_n_parents' not in dct:
            raise MissingPackingVal(argname='xov_n_parents')
        if cls.DEFAULT_varlen_okay is None and 'xov_varlen_okay' not in dct:
            raise MissingPackingVal(argname='xov_varlen_okay')
        return super().unpack(dct)
