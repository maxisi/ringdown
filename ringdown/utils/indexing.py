__all__ = ['construct_mode_list', 'construct_mode_coordinates', 
           'ModeIndex', 'ModeIndexList', 'GenericIndex']

from .. import utils
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from ast import literal_eval

def get_mode_label(mode, **kws):
    return construct_mode_index(mode).get_label(**kws)

def get_mode_coordinate(mode, **kws):
    return construct_mode_index(mode).get_coordinate()

def construct_mode_coordinates(modes : int | list) -> list:
    """Construct mode indices for InferenceData object.

    Arguments
    ---------
    modes : int or list
        Number of modes or list of mode indices.

    Returns
    -------
    idxs : list
        List of mode indices.
    """
    if isinstance(modes, ModeIndexList):
        return modes.get_coordinates()
    try:
        idxs = list(range(int(modes)))
    except TypeError:
        try:
            mode_list = [GenericIndex(m) for m in modes]
            idxs = [m.get_coordinate() for m in mode_list]
        except Exception:
            raise ValueError(f"Could not parse modes: {modes}")
    return idxs

def construct_mode_index(*mode):
    if len(mode) == 1:
        try:
            return GenericIndex(int(mode[0]))
        except (ValueError, TypeError):
            pass
    return ModeIndex.construct(*mode)

def construct_mode_list(modes : str | None) -> list:
    return ModeIndexList(modes)

class ModeIndexList(object):
    def __init__(self, indices=None):
        if indices is None:
            self.indices = []
        elif isinstance(indices, ModeIndexList):
            self.indices = indices.indices
        else:
            try:
                indices = int(indices)
                self.indices = [construct_mode_index(m) 
                                for m in range(indices)]
            except (ValueError, TypeError):
                if isinstance(indices, str):
                    # assume modes is a string like
                    # "(p0,s0,l0,m0,n0),(p1,s1,l1,m1,n1)"
                    indices = literal_eval(indices)
                self.indices = [construct_mode_index(m) for m in indices]
    
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
        return self.indices.index(construct_mode_index(x))
    
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

class ModeIndexBase(ABC):
    @abstractmethod
    def get_label(self, **kws):
        pass
    
    @abstractmethod
    def get_coordinate(self):
        pass
    
    @classmethod
    @abstractmethod
    def construct(cls):
        pass
    
    @property
    @abstractmethod
    def is_prograde(self):
        pass
    
    def as_dict(self):
        return asdict(self)

@dataclass
class GenericIndex(ModeIndexBase):
    i : int
 
    def __str__(self):
        return str(self.i)
    
    def __repr__(self):
        return f'GenericIndex(i={self.i})'
    
    def __iter__(self):
        # Yield each item one by one, making this class iterable
        for k in [self.i]:
            yield k
            
    def __int__(self) -> int:
        return self.i
    
    @property
    def is_prograde(self):
        if isinstance(self.i, ModeIndex):
            return self.i.is_prograde
        else:
            return True

    def get_label(self, **kws):
        return str(self.i)
        
    def get_coordinate(self):
        return int(self.i)
    
    @classmethod
    def construct(cls, i):
        return cls(i)
    
@dataclass
class ModeIndex(ModeIndexBase):
    p : int
    s : int
    l : int
    m : int
    n : int
    
    _keys = ('p', 's', 'l', 'm', 'n')
    
    def __iter__(self):
        # Yield each item one by one, making this class iterable
        for k in self._keys:
            yield getattr(self, k)
    
    def as_dict(self):
        return {k: getattr(self, k) for k in self._keys}
    
    def __getitem__(self, i):
        if isinstance(i, int):
            return getattr(self, self._keys[i])
        else:
            return getattr(self, i)
    
    def get_coordinate(self):
        return self.to_bytestring()
    
    def get_label(self, **kws):
        return self.to_label(**kws)

    @classmethod
    def from_string(cls, string):
        if ',' in string:
            p, s, l, m, n = map(int, string.split(','))
            return cls(p, s, l, m, n)
        else:
            # Try to parse old-style lmn strings:
            idxs = utils.string_to_tuple(string)
            if len(idxs) == 3:
                logging.warning("Assuming prograde and spin weight -2 "
                                f"for mode index: {string}; use tuple mode "
                                "indexing (p,s,l,m,n) to suppress this warning.")
                l, m, n = idxs
                p, s = 1, -2
            elif len(idxs) == 4:
                logging.warning("Assuming spin weight -2 for mode index: "
                                f"{string}; use tuple mode indexing (p,s,l,m,n)"
                                " to suppress this warning.")
                p, l, m, n = idxs
                s = -2
            elif len(idxs) == 5:
                p, s, l, m, n = idxs
            else:
                raise ValueError(f"Could not parse mode index: {string}")
            return cls(p, s, l, m, n)
        
    @property
    def is_prograde(self):
        return self.p == 1
    
    @classmethod
    def from_bytestring(cls, s):
        return cls.from_string(s.decode('utf-8'))
    
    @classmethod
    def construct(cls, *s):
        if len(s) == 1:
            s = s[0]
        if isinstance(s, ModeIndex):
            return s
        elif isinstance(s, bytes):
            return cls.from_bytestring(s)
        elif isinstance(s, str):
            return cls.from_string(s)
        else:
            return cls(*s)

    def to_bytestring(self):
        s = f'{self.p},{self.s},{self.l},{self.m},{self.n}'
        return bytes(s, 'utf-8')
    
    def to_label(self, label_prograde=False, label_spinweight=False, **kws):
        s = f'{self.l}{self.m}{self.n}'
        if label_spinweight:
            s = f'{self.s}{s}'
        if label_prograde:
            s = f'{self.p}{s}'
        return s
