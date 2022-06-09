from ...basics import basicComponent

class basicSelector(basicComponent):

    __slots__ = ('parents_needed', 'track_n_sel', 'maximize')

    DEFAULT_parents_needed = None
    DEFAULT_track_n_sel = True
    DEFAULT_maximize = None

    def __init__(self, **kargs):

        self.parents_needed, self.track_n_sel, self.maximize = None, None, None

        super().__init__(**kargs)

        if self.__class__ == basicSelector:
            self.set_params(**kargs)

    # Returns a list of selected parents
    def select_parents(self, poplst, n=2):
        raise NotImplementedError

    def set_parents_needed(self, parents_needed):
        if not isinstance(parents_needed, int):
            if isinstance(parents_needed, float) and parents_needed.is_integer():
                parents_needed = int(parents_needed)
            else:
                raise TypeError('Expected int for parents needed')
        if parents_needed < 1:
            raise ValueError('Parents Needed should be >= 1')
        self.parents_needed = parents_needed

    def get_parents_needed(self):
        if self.parents_needed is None:
            raise MissingValue('Missing parents_needed')
        return self.parents_needed


    def set_track_n_sel(self, track_n_sel):
        if not isinstance(track_n_sel, bool):
            if isinstance(track_n_sel, int):
                track_n_sel = self.int_to_bool(track_n_sel)
            else:
                raise TypeError('Expected boolean')
        self.track_n_sel = track_n_sel

    def get_track_n_sel(self):
        return self.track_n_sel

    def set_maximize(self, maximize):
        if not isinstance(maximize, bool):
            if isinstance(maximize, int):
                maximize = self.int_to_bool(maximize)
            else:
                raise TypeError('Expected boolean')
        self.maximize = maximize

    def get_maximize(self):
        return self.maximize


    def set_params(self, **kargs):

        if 'parents_needed' in kargs:
            self.set_parents_needed(kargs.get('parents_needed'))

        if 'track_n_sel' in kargs:
            self.set_track_n_sel(kargs.get('track_n_sel'))
        elif self.track_n_sel is None and ('track_n_sel' in self.config or \
                                        self.DEFAULT_track_n_sel is not None):
            self.set_track_n_sel(self.config.get('track_n_sel', \
                                        self.DEFAULT_track_n_sel, dtype=bool))

        if 'maximize' in kargs:
            self.set_maximize(kargs.get('maximize'))
        elif self.maximize is None and ('maximize' in self.config or \
                                        self.DEFAULT_maximize is not None):
            self.set_maximize(self.config.get('maximize', \
                                        self.DEFAULT_maximize, dtype=bool))



        super().set_params(**kargs)

    def pack(self, **kargs):
        dct = super().pack(**kargs)

        if kargs.get('incl_defs', False):
            dct.update({'parents_needed':self.parents_needed, \
                        'track_n_sel':self.track_n_sel})
            if self.maximize is not None:
                dct['maximize'] = self.maximize
        else:
            if self.DEFAULT_parents_needed != self.parents_needed:
                dct['parents_needed'] = self.parents_needed
            if self.DEFAULT_track_n_sel != self.track_n_sel:
                dct['track_n_sel'] = self.track_n_sel
            if self.maximize is not None and self.DEFAULT_maximize != self.maximize:
                dct['maximize'] = self.maximize


    @classmethod
    def unpack(cls, dct):
        for argname in ('parents_needed', 'track_n_sel'):
            if argname not in parents_needed:
                raise MissingPackingVal(argname=argname)
        return super().unpack(dct)
