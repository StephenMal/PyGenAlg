__version__ = '0.0.0'
__author__ = 'Stephen Maldonado'
__url__ = 'https://github.com/StephenMal/PyGenAlg'

from .optimizer import *
from .basics.basicComponent import basicComponent
from .exceptions import MissingPackingVal

def unpack_component(dct):
    if 'pygenalg.components' not in sys.modules:
        import pygenalg.components
    if 'component' not in dct:
        raise MissingPackingVal('component')
    cls = getattr(sys.modules['pygenalg.components'], dct['component'])
    return cls.unpack(dct)
