from .basics.basicSelector import basicSelector
import numpy as np
import sys, random

try:
    import numba as nb
    from numba import prange
except:
    pass

class tournamentSelector(basicSelector):

    DEFAULT_win_chance = 0.9
    DEFAULT_parents_needed = 10

    __slots__ = ('win_chance',)

    def __init__(self, **kargs):

        self.win_chance = None

        super().__init__(**kargs)

        if self.__class__ == tournamentSelector:
            self.set_params(**kargs)

    if 'numba' in sys.modules:
        @staticmethod
        @nb.jit(nopython=True, parallel=True)
        def _select_parents(fit_lst, size, winchance, n_sel, maximize):
            parents = np.full(n_sel, -1)
            indxs = np.arange(len(fit_lst))
            for n in prange(n_sel):
                tourn = np.random.choice(indxs, size=size, replace=False)

                # Get the tournament people's fitnesses
                tourn_fits = np.zeros(size)
                for t_indx in tourn:
                    tourn_fits[t_indx] = fit_lst[t_indx]

                # Organize the indicies in sorted order
                # Reverse the sorting if maximizing is more important
                if maximize:
                    tourn_argsort = np.flip(np.argsort(tourn_fits))
                else:
                    tourn_argsort = np.argsort(tourn_fits)

                # Get random values as we go through the list and try to grab
                #   the correct one
                for indx in range(size):
                    if np.random.rand() <= winchance:
                        parents[n] = tourn_argsort[indx]
                        break

                # If nothing set, pick the worst individual (low chance)
                if parents[n] == -1:
                    parents[n] = np.random.choice(tourn_argsort)

            return parents


        def select_parents(self, poplst, n=2):
            selected_indx = self._select_parents(\
                                np.array([indv.get_fit() for indv in poplst]),\
                                self.get_parents_needed(),self.get_win_chance(),\
                                n, self.get_maximize())
            lst = [poplst[indx] for indx in selected_indx]
            if self.get_track_n_sel():
                for indv in lst:
                    indv.incr_attr('n_sel', 1)
            return lst
    else:

        def select_parents(self, poplst, n=2):

            fits = [indv.get_fit() for indv in poplst]
            tourn_size = self.get_parents_needed()
            win_chance = self.get_win_chance()
            maximize = self.get_maximize()
            track_n_sel = self.get_track_n_sel()


            if win_chance == 1:
                if maximize:
                    lst = [max(random.sample(poplst, k=tourn_size)) \
                                for child in range(n)]
                else:
                    lst = [min(random.sample(poplst, k=tourn_size)) \
                                for child in range(n)]
                if track_n_sel:
                    for indv in lst:
                        indv.incr_attr('n_sel', 1)
                return lst
            else:
                # Create tournaments
                tourns = \
                    [random.sample(poplst, k=tourn_size) for i in range(n)]

                lst = [None]*n
                for p, tourn in enumerate(tourns):
                    # Sort by fitness
                    tourn = sorted(tourn, key=lambda x: x.fit, reverse=maximize)
                    # Go through until a winner wins
                    for indv in tourn:
                        if random.random() < win_chance:
                            winner[p] = indv
                            break
                    # If no one won, pick random
                    if lst[p] is None:
                        lst[p] = random.choice(tourn)
                return lst

    def set_win_chance(self, win_chance):
        if not isinstance(win_chance, float):
            if isinstance(win_chance, int):
                win_chance = float(win_chance)
            else:
                raise TypeError('Expected float for win_chance')
        if not win_chance > 0:
            raise ValueError('win_chance must be greater than 0')
        if not win_chance <= 1:
            raise ValueError('win_chance must be less than or equal to 1')
        self.win_chance = win_chance

    def get_win_chance(self):
        return self.win_chance

    def set_params(self, **kargs):
        # Set the tournament size
        if 'parents_needed' in kargs or 'tourn_size' in kargs:
            self.set_parents_needed(kargs.get('parents_needed', \
                                                    kargs.get('tourn_size')))
        elif self.parents_needed is None and 'parents_needed' not in kargs:
            self.set_parents_needed(self.config.get('tourn_size', 10,\
                                                            dtype=int, mineq=2))

        if 'parents_needed' in kargs or 'tourn_size' in kargs:
            self.set_parents_needed(kargs.get('parents_needed', \
                                                    kargs.get('tourn_size')))
        elif self.parents_needed is None and ('tourn_size' in self.config or \
                                    self.DEFAULT_parents_needed is not None):
            self.set_parents_needed(self.config.get('tourn_size', \
                                        self.DEFAULT_parents_needed,\
                                        dtype=int, mineq = 1))


        # Makes sure it is not called twice
        if 'tourn_size' in kargs and 'parents_needed' in kargs:
            if kargs.get('parents_needed') != kargs.get('tourn_size'):
                raise ValueError('Cannot pass parents_needed and tourn_size')
            kargs.pop('tourn_size')

        if 'parents_needed' in kargs:
            kargs = {key:item for key, item in kargs.items() \
                                                    if key!='parents_needed'}

        # Set win chance
        if 'tourn_win_chance' in kargs:
            self.set_win_chance(kargs.get('tourn_win_chance'))
        elif self.win_chance is None and ('tourn_win_chance' in self.config or \
                                        self.DEFAULT_win_chance is not None):
            self.set_win_chance(self.config.get('tourn_win_chance', \
                                            self.DEFAULT_win_chance,\
                                            dtype=float, min=0.0, maxeq=1.0))

        super().set_params(**kargs)

    def pack(self, **kargs):

        dct = super().pack(**kargs)

        # Win chance
        if kargs.get('incl_defs', False) or \
                            self.win_chance != self.DEFAULT_win_chance:
            dct['win_chance'] = self.win_chance

        return dct
