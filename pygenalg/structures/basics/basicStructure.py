from ...basics import basicComponent
from tempfile import TemporaryFile, NamedTemporaryFile
import sys

from vericfg import config as vcfg

# PyGenAlg module imports
import pygenalg.evaluators
import pygenalg.selectors
import pygenalg.genetic_operators

# Import basics
from ...populations.basics.basicPopulation import basicPopulation
from ...evaluators.basics.basicEvaluator import basicEvaluator
from ...genetic_operators.basics.basicOperator import basicOperator
from ...selectors.basics.basicSelector import basicSelector


class basicStructure(basicComponent):

    __slots__ = ('tempfol')

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def get_selector(self, selector, config=None):

        if isinstance(selector, str):
            if not selector[-8:].lower() == 'selector':
                selector = selector + 'Selector'
            elif selector[-8] == 's':
                selector[-8] = 'S'
            try:
                selector = getattr(sys.modules['pygenalg.selectors'], selector)
            except AttributeError:
                raise AttributeError('Invalid selector')

        if not isinstance(selector, basicSelector):
            if issubclass(selector, basicSelector):
                if config is None:
                    return selector(config=self.config)
                elif isinstance(config, (dict, vcfg)):
                    return selector(config=config)
                else:
                    raise TypeError('Expected config obj')
            else:
                raise TypeError

    def get_evaluator(self, evaluator, config=None):
        if isinstance(evaluator, str):
            if not evaluator[-9:].lower() == 'evaluator':
                evaluator = evaluator + 'Evaluator'
            elif evaluator[-9] == 'e':
                evaluator[-9] = 'E'
            try:
                evaluator = getattr(sys.modules['pygenalg.evaluators'], evaluator)
            except AttributeError:
                raise AttributeError('Invalid evaluator')

        if not isinstance(evaluator, basicEvaluator):
            if issubclass(evaluator, basicEvaluator):
                if config is None:
                    return evaluator(config=self.config)
                elif isinstance(config, (dict, vcfg)):
                    return evaluator(config=config)
                else:
                    raise TypeError('Expected config obj')
            else:
                raise TypeError

    def get_population(self, population, config=None):
        if isinstance(population, str):
            if not population[-10:].lower() == 'population':
                population += 'Population'
            elif population[-10] == 'p':
                population[-10] = 'P'
            try:
                population = getattr(sys.modules['pygeanlg.populations'], population)
            except AttributeError:
                raise AttributeError('Invalid population')

        if not isinstance(population, basicPopulation):
            if issubclass(population, basicPopulation):
                if config is None:
                    return population(config=self.config)
                elif isinstance(config, (dict, vcfg)):
                    return population(config=config)
                else:
                    raise TypeError('Expected config obj')
            else:
                raise TypeError

    def get_genetic_operator(self, gen_op, config=None):
        if isinstance(gen_op, str):
            if not gen_op[-8:].lower() == 'operator':
                gen_op += 'Operator'
            elif gen_op[-8] == 'o':
                gen_op[-8] = 'O'
            try:
                gen_op = getattr(sys.modules['pygenalg.genetic_operators'], gen_op)
            except AttributeError:
                raise AttributeError('Invalid genetic operator')

        if not isinstance(gen_op, basicOperator):
            if issubclass(gen_op, basicOperator):
                if config is None:
                    return gen_op(config=self.config)
                elif isinstance(config, (dict, vcfg)):
                    return gen_op(config=config)
                else:
                    raise TypeError('Expected config obj')
            else:
                raise TypeError
    # Run the structure
    def run(self):
        raise NotImplementedError()

    # Clears the components of the structure
    def clear(self):
        raise NotImplementedError()
