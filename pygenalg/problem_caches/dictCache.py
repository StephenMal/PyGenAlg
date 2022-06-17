from .basics.basicCache import basicCache

class dictCache(basicCache):

    __slots__ = ('dct')

    def __init__(self, *args, **kargs):

        self.dct = {}

    def get_num_solutions(self):
        return len(self.dct)

    def get(self, indv):
