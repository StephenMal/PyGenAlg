from .basics import basicOptimizer
from pygenalg.structures.basics import basicStructure
import pygenalg.structures
import sys

class geneticAlgorithm(basicOptimizer):

    __slots__ = ()

    def __init__(self, **kargs):

        super().__init__(**kargs)

    # Finds the structure and runs the optimizer
    def run(self, *args, **kargs):

        self.log.info('Getting structure')

        if 'config' in kargs:
            self.set_config(kargs.get('config'))

        struct_cls = self.config.get('struct_cls', 'generationStructure',\
                                        dtype=(str, basicStructure))

        # Turn string into struct class
        if isinstance(struct_cls, str):
            if struct_cls[-9:].lower() != 'structure':
                struct_cls = struct_cls + 'Structure'
            elif struct_cls[-9] == 's':
                struct_cls[9] = 'S'
            try:
                struct_cls = getattr(sys.modules['pygenalg.structures'], struct_cls)
            except AttributeError:
                raise AttributeError('Invalid Structure')

        # Verify a subclass of basicStructure
        if not issubclass(struct_cls, basicStructure):
            raise TypeError('Expected basicStructure class')

        # Create the structure
        struct = struct_cls(config=self.config)

        # Return the structures' results
        return struct.run()
