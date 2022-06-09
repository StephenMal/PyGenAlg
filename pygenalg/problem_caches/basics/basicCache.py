from ...basics import basicComponent


class basicCache(basicComponent):

    __slots__ = ()
    
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

        if self.__class__ == basicCache:
            self.set_params(**kargs)

    def get_num_solutions(self, *args, **kargs):
        raise NotImplementedError

    def get_hash(self, indv):
        raise NotImplementedError

    def get(self, *args, **kargs):
        raise NotImplementedError

    def set(self, *args, **kargs):
        raise NotImplementedError

    def pack(self, **kargs):
        return super().pack()

    def unpack(self, **kargs):
        return super().unpack()
