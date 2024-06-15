__all__ = ['ModeIndex', 'ModeIndexList', 'GenericIndex', 'get_mode_label',
           'get_mode_coordinate']

from . import utils
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from ast import literal_eval
from copy import copy


def get_mode_label(mode, **kws):
    return ModeIndex.construct(mode).get_label(**kws)


def get_mode_coordinate(mode, **kws):
    return ModeIndex.construct(mode).get_coordinate()


class ModeIndex(ABC):
    """Abstract class to construct mode indices. Use the `construct` method to
    create a mode index object from a string, tuple, or other object.
    """

    @abstractmethod
    def get_label(self, **kws):
        pass

    @abstractmethod
    def get_coordinate(self):
        pass

    @property
    @abstractmethod
    def is_prograde(self):
        pass

    def as_dict(self):
        return asdict(self)

    @classmethod
    def construct(cls, *mode):
        if len(mode) == 1:
            if isinstance(mode[0], ModeIndex):
                return copy(mode[0])
            try:
                return GenericIndex(int(mode[0]))
            except (ValueError, TypeError):
                pass
        return HarmonicIndex.construct(*mode)


@dataclass
class GenericIndex(ModeIndex):
    """Generic mode index for non-harmonic modes (a wrapper around an integer).
    """
    i: int

    def __eq__(self, other):
        if isinstance(other, GenericIndex):
            return self.i == other.i
        else:
            return False

    def __str__(self):
        return str(self.i)

    def __repr__(self):
        return f'GenericIndex(i={self.i})'

    def __iter__(self):
        yield self.i

    def __int__(self) -> int:
        return self.i

    @property
    def is_prograde(self):
        return True

    def get_label(self, **kws):
        return str(self.i)

    def get_coordinate(self):
        return int(self.i)

    @classmethod
    def construct(cls, i):
        return cls(i)


@dataclass
class HarmonicIndex(ModeIndex):
    """A quasinormal mode index (p, s, l, m, n), where p indicates prograde
    (1) or retrograde (-1) modes, s is the spin weight, l and m are
    the angular (spheroidal-harmonic) indices, and n is the overtone number.
    """
    p: int
    s: int
    l: int
    m: int
    n: int

    _keys = ('p', 's', 'l', 'm', 'n')

    def __iter__(self):
        # Yield each item one by one, making this class iterable
        for k in self._keys:
            yield getattr(self, k)

    def __getitem__(self, i) -> int:
        if isinstance(i, int):
            return getattr(self, self._keys[i])
        else:
            return getattr(self, i)

    def __eq__(self, other):
        if isinstance(other, HarmonicIndex):
            return all([getattr(self, k) == getattr(other, k)
                        for k in ['p', 's', 'l', 'm', 'n']])
        else:
            return False

    def as_dict(self) -> bool:
        return {k: getattr(self, k) for k in self._keys}

    @property
    def is_prograde(self) -> bool:
        return self.p == 1

    @classmethod
    def from_string(cls, string: str):
        """Construct a mode index from a string.

        Arguments
        ---------
        string : str
            string of the form 'p,s,l,m,n' or 'pslmn'.
        """
        if ',' in string:
            p, s, l, m, n = map(int, string.split(','))
            return cls(p, s, l, m, n)
        else:
            # Try to parse old-style lmn strings:
            idxs = utils.string_to_tuple(string)
            if len(idxs) == 3:
                logging.warning("Assuming prograde and spin weight -2 "
                                f"for mode index: {string}; use tuple mode "
                                "index (p,s,l,m,n) to suppress this warning.")
                l, m, n = idxs
                p, s = 1, -2
            elif len(idxs) == 4:
                logging.warning("Assuming spin weight -2 for mode index: "
                                f"{string}; use tuple mode index (p,s,l,m,n)"
                                " to suppress this warning.")
                p, l, m, n = idxs
                s = -2
            elif len(idxs) == 5:
                p, s, l, m, n = idxs
            else:
                raise ValueError(f"Could not parse mode index: {string}")
            return cls(p, s, l, m, n)

    @classmethod
    def from_bytestring(cls, s: bytes):
        return cls.from_string(s.decode('utf-8'))

    @classmethod
    def construct(cls, *s):
        """Construct a black hole mode index from a string, tuple, or other.
        Can be called as:

        construct(p, s, l, m, n)
        construct((p, s, l, m, n))
        construct('p,s,l,m,n')
        construct('pslmn')
        construct(bytes('p,s,l,m,n', 'utf-8'))

        It also accepts the following deprecated forms:

        construct('lmn')

        """
        if len(s) == 1:
            s = s[0]
        if isinstance(s, cls):
            return s
        elif isinstance(s, bytes):
            return cls.from_bytestring(s)
        elif isinstance(s, str):
            return cls.from_string(s)
        else:
            return cls(*s)

    def to_bytestring(self):
        """Convert the mode index to a bytestring."""
        s = f'{self.p},{self.s},{self.l},{self.m},{self.n}'
        return bytes(s, 'utf-8')

    def get_coordinate(self):
        """Get coordinate to use in InferenceData indexing."""
        return self.to_bytestring()

    def get_label(self, label_prograde=False, label_spinweight=False, **kws):
        """Get a string label for the mode index.

        Arguments
        ---------
        label_prograde : bool
            Include the prograde/retrograde label (default False).
        label_spinweight : bool
            Include the spin weight label (default False).
        **kws : dict
            Additional keyword arguments (ignored).
        """
        s = f'{self.l}{self.m}{self.n}'
        if label_spinweight:
            s = f'{self.s}{s}'
        if label_prograde:
            s = f'{self.p}{s}'
        return s

    def get_kerr_mode(self, **kws):
        """Get a KerrMode object for this mode index."""
        from . import qnms
        return qnms.KerrMode(self)


class ModeIndexList(object):
    def __init__(self, indices=None):
        if indices is None:
            self.indices = []
        elif isinstance(indices, ModeIndexList):
            self.indices = indices.indices
        else:
            try:
                # first look for a single integer indicating a number of modes
                indices = range(int(indices))
            except (ValueError, TypeError):
                # assume an explicit list of indices was provided in some form
                if isinstance(indices, str):
                    # assume modes is a string like
                    # "(p0,s0,l0,m0,n0),(p1,s1,l1,m1,n1)"
                    indices = literal_eval(indices)
            self.indices = [ModeIndex.construct(m) for m in indices]

    def __repr__(self):
        return f'ModeIndexList(indices={self.indices})'

    def __str__(self):
        if self.is_generic:
            return str(self.n_modes)
        else:
            return str([tuple(m) for m in self.indices])

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __getitem__(self, i):
        return self.indices[i]

    def index(self, x):
        return self.indices.index(ModeIndex.construct(x))

    @property
    def n_modes(self):
        return len(self)

    @property
    def value(self):
        if self.is_generic:
            return self.n_modes
        else:
            return [tuple(m) for m in self.indices]

    @property
    def is_generic(self):
        return all([isinstance(m, GenericIndex) for m in self.indices])

    def get_coordinates(self):
        return [m.get_coordinate() for m in self.indices]

    def get_labels(self, **kws):
        return [m.get_label(**kws) for m in self.indices]
