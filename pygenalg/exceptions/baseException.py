class PyGenAlgException(Exception):
    __slots__ = 'msg',
    pass

class MissingValue(PyGenAlgException):
    pass

class MissingPackingVal(MissingValue):

    def __init__(self, argname=None, desc=None):

        if argname is None:
            self.msg = f'Missing a critical value to pack/unpack'
        else:
            self.msg = f'Missing a critical ({argname}) value to pack/unpack'

        if desc is not None:
            self.msg += f'\n{desc}'

        super().__init__(self.msg)

class LengthError(PyGenAlgException):
    pass
