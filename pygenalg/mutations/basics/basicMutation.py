from ...basics import basicComponent
from ...exceptions import *

try:
    import numba as nb
except:
    pass

class basicMutation(basicComponent):

    __slots__ = ('mutrate')

    use_nbjit_fxn = False
    DEFAULT_mutrate = None

    def __init__(self, **kargs):

        self.mutrate = None

        super().__init__(**kargs)

        if self.__class__ == basicMutation:
            self.set_params(**kargs)

    def set_mututation_rate(self, mutrate):
        if not isinstance(mutrate, float):
            if isinstance(mutrate, int):
                mutrate = float(mutrate)
            else:
                raise TypeError('mutrate should be a float')
        if mutrate < 0 or mutrate > 1:
            raise ValueError('mutrate should be between 0.0 and 1.0')
        self.mutrate = mutrate
    # Alais
    set_mutrate = set_mututation_rate

    def get_mutrate(self):
        if self.mutrate is None:
            raise MissingValue('Missing mutrate')
        return self.mutrate

    def set_params(self, **kargs):

        if 'mutrate' in kargs:
            self.set_mututation_rate(kargs.get('mutrate'))
        elif self.mutrate is None and ('mutrate' in self.config or \
                                        self.DEFAULT_mutrate is not None):
            self.set_mututation_rate(self.config.get('mutrate', \
                                    self.DEFAULT_mutrate, mineq=0.0, maxeq=1.0,\
                                    dtype=(float, int)))

    @staticmethod
    def _mutate(chromo, mutrate):
        raise NotImplementedError

    # Applies mutation directly to an individual
    def mutate_chromo(self, item):
        if not isinstance(item, np.ndarray):
            if isinstance(item, (basicIndividual, basicChromosome)):
                item = item.to_numpy(make_copy=False)
            else:
                raise TypeError('Expected indv or basicChromosome')
        self._mutate(item.to_numpy(make_copy=False))

    # Applies mutations directly to individuals
    def mutate_batch(self, batch, **kargs):
        if self.runsafe:
            if not isinstance(batch, list):
                raise TypeError('Expected list of indvs (not a list)')
            if not all([isinstance(indv, (basicChromosome, basicIndividual)) \
                                                            for indv in batch]):
                raise TypeError('Expected list of indvs (non indv in list)')
        for indv in [indv.to_numpy(make_copy=False) \
                        if isinstance(indv, (basicChromosome, basicIndividual))\
                            else indv \
                        for indv in batch]:
            self._mutate_chromo(indv)

    def pack(self, **kargs):
        dct = super().pack(**kargs)
        dct['mutrate'] = self.mutrate
        return dct
