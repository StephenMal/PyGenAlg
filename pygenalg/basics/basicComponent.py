from vericfg import config
from psuedologger import psuedologger
import logging
from tempfile import TemporaryFile, NamedTemporaryFile
from collections import namedtuple
from ..exceptions import *
import sys, random
import numpy as np

class basicComponent():

    __slots__ = \
        {'config':'Configuration object or dictstoring user\'s parameters ',\
         'log':'Logger object (or similar object)',\
         'runsafe':'Whether or not to perform extra validation checks (for debug)'}

    DEFAULT_config = {}
    DEFAULT_log = psuedologger()
    DEFAULT_runsafe = True

    nprng = None    # Numpy random generator
    rseed = None    # Python's random seed

    # Stores the python version
    pyversion = sys.version_info

    # Lists given warnings to prevent repeats
    warnings_given = set()

    # Special case, variables set in __init__ instead of set_params()
    def __init__(self, **kargs):

        # Set as None
        self.config, self.log, self.runsafe = None, None, None

        #
        if 'config' in kargs:
            self.set_config(kargs.get('config'))
        elif self.config is None and self.DEFAULT_config is not None:
            self.set_config(self.DEFAULT_config)

        if 'log' in kargs:
            self.set_log(kargs.get('log'))
        elif 'logger' in kargs:
            self.set_log(kargs.get('logger'))
        elif self.log is None and \
                        ('log' in self.config or self.DEFAULT_log is not None):
            if 'log' in self.config:
                self.set_log(self.config.get('log', self.DEFAULT_log))
            elif 'logger' in self.config:
                self.set_log(self.config.get('logger', self.DEFAULT_log))
            else:
                self.set_log(self.config.get('log',self.DEFAULT_log))

        if 'runsafe' in kargs:
            self.set_runsafe(kargs.get('runsafe'))
        elif self.runsafe is None:
            self.set_runsafe(self.config.get('runsafe', \
                                            self.DEFAULT_runsafe, dtype=bool))

        if self.rseed is None and 'rseed' in kargs:
            self.set_rseed(kargs.get('rseed'))
        elif self.rseed is None:
            self.set_rseed(self.config.get('rseed',random.randint(1,99999999),\
                                                        dtype=int, \
                                                        mineq=1, max=99999999))

        # Print debug this object has been created
        self.log.debug(f'Initializing {self.__class__.__name__} object')

    def set_params(**kargs):
        pass

    def set_config(self, cfg):
        if not isinstance(cfg, config):
            if isinstance(cfg, dict):
                cfg = config(cfg)
            else:
                raise TypeError('Expected config obj or dict')
        self.config = cfg

    def set_log(self, log):
        if not isinstance(log, psuedologger):
            try:
                log.debug('Testing logger')
            except:
                raise Exception('Log needs to support similar function '+\
                                'calls to python\'s built in logger')
        self.log = log

    def set_runsafe(self, runsafe):
        if not isinstance(runsafe, bool):
            if isinstance(runsafe, int):
                runsafe = self.int_to_bool(runsafe)
            else:
                raise TypeError('Expected bool for runsafe')
        self.runsafe = runsafe

    @classmethod
    def set_rseed(cls, rseed, overwrite=False):
        if not isinstance(rseed, int):
            if isinstance(rseed, float) and rseed.is_integer():
                rseed = int(rssed)
            else:
                raise TypeError('Expected int for rseed')
        if cls.rseed is None or overwrite:
            # Save rseed
            cls.rseed = rseed
            # Set the python rseed and the nprng rseed
            random.seed(rseed)
            cls.nprng = np.random.default_rng(rseed)
        else:
            self.pga_warning('Attempted to overwrite rseed')

    def set_params(self, **kargs):
        return

    # Deletes everything
    def __del__(self):
        self.config, self.log, self.runsafe = None, None, None

    @staticmethod
    def verify_version(major=None, minor=None, micro=None, rl=None, serial=None):
        if major is None and minor is None and micro is None and rl is None and\
                            serial is None:
            raise ValueError('No input')
        pyv = sys.version_info
        return ((major is None or major >= pyv.major) and \
                (minor is None or minor >= pyv.minor) and \
                (micro is None or micro >= pyv.micro) and \
                (rl is None or rl == pyv.rl) and \
                (serial is None or serial >= pyv.serial))

    # Returns a TemporaryFile
    def create_temp_file(self, *args, **kargs):
        # Verify errors is not used before python 3.8
        if self.verify_version(3,8) and 'errors' in kargs:
            self.pga_warning('errors cannot be used in create_temp_file before 3.8')
            kargs.pop('errors')
        return TemporaryFile(*args, **kargs)

    # Returns a NamedTemporyFile
    def create_named_temp_file(self, *args, **kargs):
        # Verify errors is not used before python 3.8
        if self.verify_version(3,8) and 'errors' in kargs:
            self.pga_warning('errors cannot be used in create_temp_file before 3.8')
            kargs.pop('errors')
        return NamedTemporaryFile(*args, **kargs)

    # Returns whether or not an item is hashable
    @staticmethod
    def _is_hashable(item):
        try:
            item.__hash__()
            return True
        except:
            return False

    # Returns whether or not an item is iterable
    @staticmethod
    def _is_iterable(item):
        try:
            iter(item)
            return True
        except:
            return False

    # Convert 0 or 1 to bool, reject all else
    @staticmethod
    def int_to_bool(num):
        if isinstance(num, int):
            if num == 0:
                return False
            elif num == 1:
                return True
            else:
                raise ValueError('Expected 0 or 1')
        elif isinstance(num, bool):
            return num
        elif num == 3:
            return None
        else:
            raise TypeError('Expected int (0 or 1) or bool')

    # Convert 0 or 1 to bool, reject all else
    @staticmethod
    def bool_to_int(boolean):
        if isinstance(boolean, bool):
            if boolean:
                return 1
            return 0
        elif isinstance(num, int):
            if num == 0 or num == 1:
                return num
            else:
                raise ValueError('num should be 0 or 1')
        elif num is None:
            return 3
        else:
            raise TypeError('Expected bool or int (0 or 1)')

    def pga_warning(self, msg):
        if msg not in self.warnings_given:
            self.log.warn(msg)
            self.warnings_given.add(msg)


    # Returns values in dictionary
    def to_dict(self, **kargs):
        return self.pack(**kargs)

    @classmethod
    def from_dict(cls, dct):
        dct = {argn:argv for argn, argv in dct.items() if argv is not None}
        return cls.unpack(dct)

    def __copy__(self):
        return self.unpack(self.pack(compress=False, incl_bsc_cmp=True))

    def pack(self, **kargs):
        if not kargs.get('incl_bsc_cmp', False):
            return {'component':self.__class__.__name__}

        dct = {'config':self.config.to_dict(), 'rseed':self.rseed}

        dct['component'] = self.__class__.__name__

        if kargs.get('incl_defs', False) or self.runsafe != self.DEFAULT_runsafe:
            dct['runsafe'] = self.bool_to_int(self.runsafe)

        if kargs.get('incl_defs', False) or \
                                        not isinstance(self.log, psuedologger):
            try:
                dct['log'] = self.log.name
            except:
                pass

        return dct

    @classmethod
    def unpack(cls, dct):
        if 'log' in dct:
            log = dct['log']
            if isinstance(log, str):
                log = logging.getLogger(log)
                dct['log'] = log
        return cls(**dct)

    # Allows unpacking of any component without knowing the actual class
    @staticmethod
    def unpack_component(dct):
        if 'pygenalg.components' not in sys.modules:
            import pygenalg.components
        if 'component' not in dct:
            raise MissingPackingVal('component')
        cls = getattr(sys.modules['components'], dct['component'])
        return cls.unpack(dct)
