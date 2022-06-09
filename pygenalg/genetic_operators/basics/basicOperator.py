from ...basics import basicComponent
from ...crossovers.basics.basicCrossover import basicCrossover
from ...mutations.basics.basicMutation import basicMutation
import pygenalg.crossovers
import pygenalg.mutations
import sys, inspect

class basicOperator(basicComponent):

    __slots__ = ('xov', 'mut')

    DEFAULT_xov = 'twoPointCrossover'
    DEFAULT_mut = 'uniformMutation'

    def __init__(self, **kargs):
        self.xov = None
        self.mut = None

        super().__init__(**kargs)

        if self.__class__ == basicOperator:
            self.set_params(**kargs)

    def create_children(self, sel_p, children_pop):
        # Verify we have crossover and mutation objects
        if self.xov is None:
            raise MissingValue('Missing xov')
        elif self.mut is None:
            raise MissingValue('Missing mut')

        # Move xov and mut to closer scope
        xov, mut = self.xov, self.mut

        xov_n_p = xov.get_n_parents_needed()

        if len(sel_p) % xov_n_p != 0:
            raise Exception('Number of parents should be evenly divisible by '+\
                            'number of parents needed for xov')

        # Create children npndarrays
        c_arrs = \
            xov.cross_batch([sel_p[i:i+n] for i in range(0,len(sel_p),xov_n_p)])

        for child, arr in zip(children_pop, c_arrs):
            child.set_chromo(arr)

        mut.mutate_batch(children_pop.to_list())

    def set_xov(self, xov):
        if not isinstance(xov, basicCrossover):
            if isinstance(xov, str):
                if len(xov) < 9 or xov[-9:].lower() != 'crossover':
                    xov = xov + 'Crossover'
                elif xov[-9] == 'c':
                    xov[-9] = 'C'
                try:
                    xov = getattr(sys.modules['pygenalg.crossovers'], xov)
                except AttributeError:
                    raise AttributeError(f'Invalid crossover:\n{xov}')
                xov = xov(config=self.config)
            elif isinstance(xov, dict):
                try:
                    xov = self.unpack_component(xov)
                except:
                    raise Exception('Failed to unpack')
            else:
                raise TypeError\
                    ('Expected crossover class or str of crossover class name')
        self.xov = xov

    def set_mut(self, mut):
        if not isinstance(mut, basicMutation):
            if isinstance(mut, str):
                if len(mut) < 8 or mut[-8:].lower() != 'mutation':
                    mut = mut + 'Mutation'
                elif mut[-8] == 'm':
                    mut[-8] = 'M'
                try:
                    mut = getattr(sys.modules['pygenalg.mutations'], mut)
                except AttributeError:
                    raise AttributeError(f'Invalid mutation:\n{mut}')
                mut = mut(config=self.config)
            elif isinstance(mut, dict):
                try:
                    mut = self.unpack_component(mut)
                except:
                    raise Exception('Failed to unpack')
            else:
                raise TypeError\
                    ('Expected mutation class or str of mutation class name')
        self.mut = mut

    def set_params(self, **kargs):

        if 'xov' in kargs:
            self.set_xov(kargs.get('xov'))
        elif self.xov is None and ('xov' in self.config or \
                                            self.DEFAULT_xov is not None):
            self.set_xov(self.config.get('xov', self.DEFAULT_xov, \
                                            dtype=(str, basicCrossover, dict)))


        if 'mut' in kargs:
            self.set_mut(kargs.get('mut'))
        elif self.mut is None and ('mut' in self.config or \
                                        self.DEFAULT_mut is not None):
            self.set_mut(self.config.get('mut', self.DEFAULT_mut, \
                                            dtype=(str, basicMutation, dict)))

    def pack(self, **kargs):

        dct = super().pack(**kargs)

        if self.xov is not None:
            dct['xov'] = self.xov.pack()
        if self.mut is not None:
            dct['mut'] = self.mut.pack()

        return dct
