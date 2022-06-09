from ...basics import basicComponent
from vericfg import config as vcfg

class basicOptimizer(basicComponent):

    __slots__ = ()

    def __init__(self, **kargs):

        super().__init__(**kargs)

    # Builds the optimizer
    def run(self, **kargs):
        raise NotImplementedError()
