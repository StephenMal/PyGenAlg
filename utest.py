import unittest, sys, logging
from copy import copy, deepcopy
from vericfg import config
from itertools import product as iterprod
from psuedologger import psuedologger
from pygenalg.exceptions import *
import os
import numpy as np


''' basicComponent '''
from pygenalg.basics.basicComponent import basicComponent
class basicComponentTest(unittest.TestCase):

    def test_init(self):
        with self.subTest(input='dict'):
            x = basicComponent(config={})
        with self.subTest(input='config'):
            x = basicComponent(config=config({}))
        with self.subTest(input='psuedologger karg = log'):
            x = basicComponent(config={'log':psuedologger()})
        with self.subTest(input='psuedologger karg = logger'):
            x = basicComponent(config={'logger':psuedologger()})
        with self.subTest(input='logger module karg = log'):
            x = basicComponent(config={'log':logging})
        with self.subTest(input='logger module karg = logger'):
            x = basicComponent(config={'logger':logging})
        with self.subTest(input='logger obj karg = log'):
            x = basicComponent(config={'log':logging.getLogger('test')})
        with self.subTest(input='logger obj karg = logger'):
            x = basicComponent(config={'logger':logging.getLogger('test')})
        with self.subTest(input='Missing config'):
            x = basicComponent()
            self.assertNotEqual(x.config, None)
        with self.subTest(input='Config incorrect dtype'):
            with self.assertRaises(TypeError):
                x = basicComponent(config=3)
        with self.subTest(input='Bad logger'):
            with self.assertRaises(TypeError):
                x = basicComponent(config={'log':3})

    def test_del(self):
        x = basicComponent(config={})
        del x

    def test_verify_version(self):
        x = basicComponent(config={})
        ma, min, mic, rl, s = sys.version_info

        for _ma in range(10):
            with self.subTest(input=f'major={_ma} but truly {ma}'):
                if _ma is None:
                    with self.assertRaises(ValueError):
                        x.verify_version(macro=_ma)
                elif _ma < ma:
                    self.assertFalse(x.verify_version(major=_ma))
                else:
                    self.assertTrue(x.verify_version(major=_ma))

        for _min in range(10):
            with self.subTest(input=f'minor={_min} but truly {_min}'):
                if _min is None:
                    with self.assertRaises(ValueError):
                        x.verify_version(minor=_min)
                elif _min < min:
                    self.assertFalse(x.verify_version(minor=_min))
                else:
                    self.assertTrue(x.verify_version(minor=_min))

        for _mic in range(10):
            with self.subTest(input=f'micro={_mic} but truly {_mic}'):
                if _mic is None:
                    with self.assertRaises(ValueError):
                        x.verify_version(micro=_mic)
                elif _min < mic:
                    self.assertFalse(x.verify_version(micro=_mic))
                else:
                    self.assertTrue(x.verify_version(micro=_mic))

        for _s in range(10):
            with self.subTest(input=f'serial={_s} but truly {_s}'):
                if _s is None:
                    with self.assertRaises(ValueError):
                        x.verify_version(serial=_s)
                elif _s < s:
                    self.assertFalse(x.verify_version(serial=_s))
                else:
                    self.assertTrue(x.verify_version(serial=_s))


        for _ma, _min, _mic, _s in iterprod(list(range(10)),list(range(10)),\
                                            list(range(10)),list(range(10))):
            with self.subTest(input=f'inp:({_ma},{_min},{_mic},{_s}), '+\
                                    f'tru:({ma},{min},{mic},{s})'):

                if _ma < ma or _min < min or _mic < mic or _s < s:
                    self.assertFalse(x.verify_version(major=_ma, minor=_min,\
                                                      micro=_mic, serial=_s))
                else:
                    self.assertTrue(x.verify_version(major=_ma, minor=_min,\
                                                     micro=_mic, serial=_s))

        del x

    def test_create_temp_file(self):
        x = basicComponent(config={})
        fp = x.create_temp_file()
        fp.write(b'hello world')
        fp.seek(0)
        self.assertTrue(fp.read() == b'hello world')
        fp.close()
        del x

    def test_hashable(self):
        x = basicComponent(config={})
        hashable_item, nonhashable_item = frozenset([1,2,3]), set([1,2,3])
        with self.subTest(input=f'is hashable'):
            self.assertTrue(x._is_hashable(hashable_item))
        with self.subTest(input=f'is not hashable'):
            self.assertFalse(x._is_hashable(nonhashable_item))

    def test_iterable(self):
        x = basicComponent(config={})
        iterable_item, noniterable_item = [1,2,3], 1
        with self.subTest(input=f'is hashable'):
            self.assertTrue(x._is_iterable(iterable_item))
        with self.subTest(input=f'is not hashable'):
            self.assertFalse(x._is_iterable(noniterable_item))

    def test_to_dict(self):
        x = basicComponent(config={})
        if sys.version_info[1] >= 4:
            with self.subTest(input='to_dict'):
                dct = x.to_dict()
                self.assertIsInstance(dct, dict)
        else:
            with self.subTest(input='to_dict'):
                dct = x.to_dict()
                self.assertEqual(type(dct), dict)
        with self.subTest(input='to_dict values'):
            dct = x.to_dict()
            self.assertEqual(dct['runsafe'], True)

    def test_from_dict(self):
        with self.subTest(input='from_dict'):
            y = basicComponent.from_dict({'config':{}, 'log':None, 'runsafe':True})
        with self.subTest(input='form basicComponent.to_dict()'):
            y = basicComponent.from_dict(basicComponent(config={}).to_dict())

    def test_copy(self):
        x = basicComponent(config={})
        with self.subTest(input='__copy__'):
            y = x.__copy__()
        with self.subTest(input='copy'):
            y = copy(x)
        with self.subTest(input='deepcopy'):
            y = deepcopy(x)

''' BasicChromosome '''

from pygenalg.indvs.basics.basicChromosome import basicChromosome
class basicChromosomeTest(unittest.TestCase):

    def test_init_del(self):
        with self.subTest('init'):
            x = basicChromosome(config={})
        with self.subTest('del'):
            del x

    def test_set_params(self):
        x = basicChromosome(config={})
        with self.subTest('dtype'):
            with self.subTest('dtype-float'):
                with self.subTest('dtype-float-str'):
                    x.set_params(dtype='float')
                    self.assertEqual(x.dtype, float)
                    self.assertEqual(x.np_dtype, np.float64)
                with self.subTest('dtype-float-type'):
                    x.set_params(dtype=float)
                    self.assertEqual(x.dtype, float)
                    self.assertEqual(x.np_dtype, np.float64)
            with self.subTest('dtype-int'):
                with self.subTest('dtype-int-str(int)'):
                    x.set_params(dtype='int')
                    self.assertEqual(x.dtype, int)
                    self.assertEqual(x.np_dtype, np.int64)
                with self.subTest('dtype-int-str(integer)'):
                    x.set_params(dtype='integer')
                    self.assertEqual(x.dtype, int)
                    self.assertEqual(x.np_dtype, np.int64)
                with self.subTest('dtype-int-type'):
                    x.set_params(dtype=int)
                    self.assertEqual(x.dtype, int)
                    self.assertEqual(x.np_dtype, np.int64)


        with self.subTest('dsize'):
            with self.subTest('dsize-int'):
                with self.subTest('dsize-int-8'):
                    x.set_params(dtype=int, dsize=8)
                    self.assertEqual(x.dtype, int)
                    self.assertEqual(x.np_dtype, np.int8)
                with self.subTest('dsize-int-16'):
                    x.set_params(dtype=int, dsize=16)
                    self.assertEqual(x.dtype, int)
                    self.assertEqual(x.np_dtype, np.int16)
                with self.subTest('dsize-int-32'):
                    x.set_params(dtype=int, dsize=32)
                    self.assertEqual(x.dtype, int)
                    self.assertEqual(x.np_dtype, np.int32)
                with self.subTest('dsize-int-64'):
                    x.set_params(dtype=int, dsize=64)
                    self.assertEqual(x.dtype, int)
                    self.assertEqual(x.np_dtype, np.int64)
            with self.subTest('dsize-float'):
                with self.subTest('dsize-float-8'):
                    with self.assertRaises(ValueError):
                        x.set_params(dtype=float, dsize=8)
                with self.subTest('dsize-float-16'):
                    x.set_params(dtype=float, dsize=16)
                    self.assertEqual(x.dtype, float)
                    self.assertEqual(x.np_dtype, np.float16)
                with self.subTest('dsize-float-32'):
                    x.set_params(dtype=float, dsize=32)
                    self.assertEqual(x.dtype, float)
                    self.assertEqual(x.np_dtype, np.float32)
                with self.subTest('dsize-float-64'):
                    x.set_params(dtype=float, dsize=64)
                    self.assertEqual(x.dtype, float)
                    self.assertEqual(x.np_dtype, np.float64)

        with self.subTest('min'):
            with self.subTest('min-int'):
                with self.subTest('min-int-pass'):
                    x.set_params(dtype=int, min=0)
                    self.assertEqual(x.dtype, int)
                    self.assertEqual(x.minv, 0)
                    with self.subTest('min-int-fxn'):
                        self.assertEqual(x.get_minv(), 0)
                with self.subTest('min-int-fail'):
                    with self.assertRaises(TypeError):
                        x.set_params(dtype=int, min=0.0)
            with self.subTest('min-float'):
                with self.subTest('min-float-pass'):
                    x.set_params(dtype=float, min=0.0)
                    self.assertEqual(x.dtype, float)
                    self.assertEqual(x.minv, 0.0)
                    with self.subTest('min-float-fxn'):
                        self.assertEqual(x.get_minv(), 0.0)
                with self.subTest('min-float-fail'):
                    with self.assertRaises(TypeError):
                        x.set_params(dtype=float, min=0)

        with self.subTest('max'):
            with self.subTest('max-int'):
                with self.subTest('max-int-pass'):
                    x.set_params(dtype=int, max=10)
                    self.assertEqual(x.dtype, int)
                    self.assertEqual(x.maxv, 10)
                    with self.subTest('max-int-fxn'):
                        self.assertEqual(x.get_maxv(), 10)
                with self.subTest('max-int-fail'):
                    with self.assertRaises(TypeError):
                        x.set_params(dtype=int, max=10.0)
            with self.subTest('max-float'):
                with self.subTest('max-float-pass'):
                    x.set_params(dtype=float, max=10.0)
                    self.assertEqual(x.dtype, float)
                    self.assertEqual(x.maxv, 10.0)
                    with self.subTest('max-float-fxn'):
                        self.assertEqual(x.get_maxv(), 10.0)
                with self.subTest('max-float-fail'):
                    with self.assertRaises(TypeError):
                        x.set_params(dtype=float, max=10)

        with self.subTest('minlen'):
            with self.subTest('minlen-wrongdtype'):
                with self.assertRaises(TypeError):
                    x.set_params(maxlen=[])
            with self.subTest('minlen-negative'):
                with self.assertRaises(ValueError):
                    x.set_params(minlen=-1)
            with self.subTest('minlen-pass'):
                x.set_params(minlen=10)
                self.assertEqual(x.minlen, 10)

        with self.subTest('maxlen'):
            with self.subTest('maxlen-wrongdtype'):
                with self.assertRaises(TypeError):
                    x.set_params(maxlen=[])
            with self.subTest('maxlen-negative'):
                with self.assertRaises(ValueError):
                    x.set_params(maxlen=-1)
            with self.subTest('maxlen-pass'):
                x.set_params(maxlen=100)
                self.assertEqual(x.maxlen, 100)

        with self.subTest('varlen'):
            with self.subTest('varlen-pass-vlen'):
                x.set_params(minlen=1, maxlen=100, varlen=True)
                self.assertTrue(x.varlen)
            with self.subTest('varlen-pass-flen'):
                x.set_params(minlen=10, maxlen=10, varlen=False)
                self.assertFalse(x.varlen)
            with self.subTest('varlen-fail-vlen'):
                with self.assertRaises(ValueError):
                    x.set_params(minlen=10, maxlen=10, varlen=True)
            with self.subTest('varlen-fail-flen'):
                with self.assertRaises(ValueError):
                    x.set_params(minlen=10, maxlen=100, varlen=False)

        with self.subTest('length'):
            with self.subTest('length-pass-vlen'):
                x.set_params(minlen=1, maxlen=100, varlen=True, length=50)
            with self.subTest('length-pass-flen'):
                x.set_params(minlen=50, maxlen=50, varlen=False, length=50)
            with self.subTest('length-fail-flen-min'):
                with self.assertRaises(ValueError):
                    x.set_params(minlen=50, maxlen=50, varlen=False, length=49)
            with self.subTest('length-fail-flen-max'):
                with self.assertRaises(ValueError):
                    x.set_params(minlen=50, maxlen=50, varlen=False, length=51)
            with self.subTest('length-fail-vlen-min'):
                with self.assertRaises(ValueError):
                    x.set_params(minlen=25, maxlen=75, varlen=True, length=24)
            with self.subTest('length-fail-vlen-max'):
                with self.assertRaises(ValueError):
                    x.set_params(minlen=25, maxlen=75, varlen=True, length=76)

        with self.subTest('vals'):
            with self.subTest('vals-list-int-minlen1-maxlen10-vlen-len3-[1,2,3]'):
                x.set_params(minlen=1, maxlen=10, varlen=True, length=3, \
                             dtype=int, vals=[1,2,3])
            with self.subTest('vals-list-int-minlen1-maxlen10-flen-len3-[1,2,3]'):
                x.set_params(minlen=3, maxlen=3, varlen=False, length=3, \
                             dtype=int, vals=[1,2,3])

    def test_search(self):
        x = basicChromosome(vals=[1,2,3])
        with self.subTest('[1,2,3] in [1,2,3]'):
            self.assertEqual(list(x.search([1,2,3])),[0])
        with self.subTest('1 in [1,2,3]'):
            self.assertEqual(list(x.search(1)),[0])
        with self.subTest('[2,3] in [1,2,3]'):
            self.assertEqual(list(x.search([2,3])),[1])
        with self.subTest('2 in [1,2,3]'):
            self.assertEqual(list(x.search(2)),[1])
        with self.subTest('[3] in [1,2,3]'):
            self.assertEqual(list(x.search([3])),[2])
        with self.subTest('3 in [1,2,3]'):
            self.assertEqual(list(x.search(3)),[2])

''' BasicIndividual '''
from pygenalg.indvs.basics.basicIndividual import basicIndividual
class basicIndividualTest(unittest.TestCase):

    def test_init_del(self):
        with self.subTest('init'):
            x = basicIndividual(config={})
            with self.subTest('del'):
                del x

    def test_set_id(self):
        with self.subTest('init id'):
            x = basicIndividual(config={}, id=0)
            self.assertEqual(x.id, 0)
        with self.subTest('set id'):
            x = basicIndividual(config={})
            x.set_id(0)
            self.assertEqual(x.id, 0)

    def test_set_fit(self):
        with self.subTest('set_fit test'):
            x = basicIndividual(config={})
            x.set_fit(10.0)
            with self.subTest('correct value'):
                self.assertEqual(x.fit, 10)
            with self.subTest('correct dtype'):
                self.assertEqual(type(x.fit), float)
            del x
        with self.subTest('init fit'):
            x = basicIndividual(config={}, fit=10.0)
            with self.subTest('init fit correct value'):
                self.assertEqual(x.fit, 10)
            with self.subTest('init fit correct dtype'):
                self.assertEqual(type(x.fit), float)
            del x
        with self.subTest('set_fit failure and resolve checking'):
            x = basicIndividual(config={})
            with self.subTest('set_fit fail/resolve - int -> float'):
                x.set_fit(10)
                with self.subTest('set_fit fail/resolve - int -> float correct value'):
                    self.assertEqual(x.fit, 10)
                with self.subTest('set_fit fail/resolve - int -> float correct dtype'):
                    self.assertEqual(type(x.fit), float)
            with self.subTest('set_fit fail/resolve - str -> float'):
                x.set_fit(np.float_(10))
                with self.subTest('set_fit fail/resolve - str -> float correct value'):
                    self.assertEqual(x.fit, 10)
                with self.subTest('set_fit fail/resolve - str -> float correct dtype'):
                    self.assertEqual(type(x.fit), float)

    def test_attrs(self):
        x = basicIndividual(config={})
        with self.subTest('verify attrs is None'):
            self.assertEqual(x.attrs, None)
        with self.subTest('set_attr'):
            x.set_attr('attr1', 0)
            with self.subTest('set_attr - verify create dict'):
                self.assertEqual(type(x.attrs), dict)
            with self.subTest('set_attr - verify value saved'):
                self.assertEqual(x.attrs['attr1'], 0)
        with self.subTest('set_attr 2'):
            x.set_attr('attr2', 2)
            with self.subTest('set_attr - verify still a dict'):
                self.assertEqual(type(x.attrs), dict)
            with self.subTest('set_attr - verify value 1 still saved'):
                self.assertEqual(x.attrs['attr1'], 0)
            with self.subTest('set_attr - verify value 2 saved'):
                self.assertEqual(x.attrs['attr2'], 2)
        with self.subTest('set_attr fail'):
            with self.assertRaises(TypeError):
                x.set_attr(4, 5)
        with self.subTest('test clear'):
            x.clear_attrs()
            self.assertEqual(x.attrs, None)
        with self.subTest('set_attrs'):
            x.set_attrs({'attr1':1, 'attr2':2})
            with self.subTest('set_attrs - Verify type'):
                self.assertEqual(type(x.attrs), dict)
            with self.subTest('set_attrs - verify value 1'):
                self.assertEqual(x.attrs['attr1'], 1)
            with self.subTest('set_attrs - verify value 2'):
                self.assertEqual(x.attrs['attr2'], 2)
        with self.subTest('get_attr'):
            x.set_attrs({'attr1':1, 'attr2':2})
            with self.subTest('get_attr - attr1'):
                self.assertEqual(x.get_attr('attr1'), 1)
            with self.subTest('get_attr - attr2'):
                self.assertEqual(x.get_attr('attr2'), 2)
        with self.subTest('get_attrs'):
            dct = {'attr1':1, 'attr2':2}
            x.set_attrs(dct)
            self.assertEqual(x.get_attrs(), dct)
        with self.subTest('update_attrs'):
            dct = {'attr1':1}
            dct2 = {'attr2':2, 'attr3':3}
            x.set_attrs(dct)
            x.update_attrs(dct2)
            with self.subTest('update_attrs - verify worked'):
                dct3 = dct2.copy()
                dct3.update(dct)
                self.assertEqual(x.get_attrs(), dct3)
            with self.subTest('update_attrs - TypeError'):
                with self.assertRaises(TypeError):
                    x.update_attrs([1,2,3])
        with self.subTest('incr_attr'):
            dct = {'attr1': 1, 'attr2': 2, 'attr3': 3}
            x.set_attrs(dct)
            self.assertEqual(x.attrs['attr1'], 1)
            x.incr_attr('attr1', 1)
            self.assertEqual(x.attrs['attr1'], 2)
        with self.subTest('del_attr'):
            dct = {'attr1': 1, 'attr2': 2, 'attr3': 3}
            x.set_attrs(dct)
            self.assertEqual(x.attrs['attr1'], 1)
            x.del_attr('attr1')
            with self.assertRaises(KeyError):
                y = x.attrs['attr1']

    def test_mapped(self):
        x = basicIndividual(config={})
        with self.assertRaises(NotImplementedError):
            x.map(None)
        x.chromo = None
        with self.assertRaises(MissingValue):
            x.get_mapped()

''' BinaryChromosome '''
from pygenalg.indvs.binary import binaryChromosome
class binaryChromosomeTest(unittest.TestCase):
    def test_init_del(self):
        with self.subTest('init'):
            x = binaryChromosome()
        with self.subTest('del'):
            del x

    def test_forced_value_check(self):
        x = binaryChromosome()
        with self.subTest('minv'):
            self.assertEqual(x.minv, 0)
        with self.subTest('maxv'):
            self.assertEqual(x.maxv, 1)
        with self.subTest('dtype'):
            self.assertEqual(x.dtype, int)
        with self.subTest('np_dtype'):
            self.assertEqual(x.np_dtype, np.uint8)

    def test_minmax(self):
        x_all0 = binaryChromosome(vals=[0,0,0])
        x_mix = binaryChromosome(vals=[0,1,0])
        x_all1 = binaryChromosome(vals=[1,1,1])

        with self.subTest('x_all0 min'):
            self.assertEqual(x_all0.min(), 0)
        with self.subTest('x_all0 max'):
            self.assertEqual(x_all0.max(), 0)
        with self.subTest('x_mix min'):
            self.assertEqual(x_mix.min(), 0)
        with self.subTest('x_mix max'):
            self.assertEqual(x_mix.max(), 1)
        with self.subTest('x_all1 min'):
            self.assertEqual(x_all1.min(), 1)
        with self.subTest('x_all1 max'):
            self.assertEqual(x_all1.max(), 1)

        del x_all0, x_mix, x_all1

    def test_search(self):
        x = binaryChromosome(vals=[0,0,0,1,1,1,0,1,0,1])
        with self.subTest('[0,1] in [0,0,0,1,1,1,0,1,0,1]'):
            self.assertEqual(list(x.search([0,1])),[2,6,8])
        with self.subTest('0 in [0,0,0,1,1,1,0,1,0,1]'):
            self.assertEqual(list(x.search(0)), [0,1,2,6,8])
        with self.subTest('[0,0,0,1,1,1,0,1,0,1] in [0,0,0,1,1,1,0,1,0,1]'):
            self.assertEqual(list(x.search([0,0,0,1,1,1,0,1,0,1])), [0])
        del x

''' BinaryIndividual '''
from pygenalg.indvs.binary import binaryIndividual
class BinaryIndividualTest(unittest.TestCase):

    def test_init_del(self):
        with self.subTest('init'):
            x = binaryIndividual()
            with self.subTest('del'):
                del x

    def test_map(self):
        pass

''' '''

if __name__ == '__main__':

    print('Starting unittest')
    psuedologger.toggle(enable=False)
    unittest.main()
    psuedologger.toggle(enable=True)
