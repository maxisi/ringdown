"""Utilities for the ringdown package.
"""

import numpy as np
from ast import literal_eval
import os
from configparser import ConfigParser
import logging
import sys


def get_tqdm(progress: bool = True):
    """Return the appropriate tqdm based on the execution environment.
    """
    if not progress:
        def custom_tqdm(args, **kwargs):
            return args
        return custom_tqdm
    if 'ipykernel' in sys.modules:
        # Running in Jupyter Notebook/Lab
        from tqdm.notebook import tqdm
    else:
        # Running in a terminal or other non-notebook environment
        from tqdm import tqdm
    return tqdm


def form_opt(x, key=None, **kws) -> str:
    """Utility to format options in config.

    Parameters
    ----------
    x : str, list, or dict
        The option to format.
    kws : dict
        Additional keyword arguments to pass to np.array2string.

    Returns
    -------
    str
        The formatted option.
    """
    if key == 't0' and 'precision' not in kws:
        kws['precision'] = 16
    return np.array2string(np.array(x), separator=', ', **kws)


def try_parse(x):
    """Attempt to parse a string as a number, dict, or list."""
    try:
        return float(x)
    except (TypeError, ValueError):
        try:
            return literal_eval(x)
        except (TypeError, ValueError, SyntaxError):
            if x == "inf":
                return np.inf
            else:
                return x


def get_ifo_list(config, section):
    ifo_input = config.get(section, 'ifos', fallback='')
    try:
        ifos = literal_eval(ifo_input)
    except (ValueError, SyntaxError):
        ifos = [i.strip() for i in ifo_input.split(',')]
    return ifos


def get_hdf5_value(x):
    """Attempt to parse a string as a number, dict, or list."""
    while isinstance(x, np.ndarray) and len(x) == 1:
        x = x[0]
    if isinstance(x, np.ndarray) and len(x) > 1:
        return [get_hdf5_value(i) for i in x]
    if isinstance(x, (bytes, np.bytes_)):
        return try_parse(x.decode('utf-8'))
    else:
        return try_parse(x)


def np2(x):
    """Returns the next power of two as big as or larger than x."""
    p = 1
    while p < x:
        p = p << 1
    return p


def isp2(x):
    """Returns True if x is a power of two."""
    is_pow2 = x == int(x)
    if is_pow2:
        x = int(x)
        is_pow2 &= x & (x - 1) == 0
    return is_pow2


def get_dict_from_pattern(path, ifos=None, abspath=False):
    if isinstance(path, str):
        path_dict = try_parse(path)
        if isinstance(path_dict, str):
            if ifos is None:
                raise ValueError("must provide IFO list.")
            path_dict = {}
            for ifo in ifos:
                i = '' if not ifo else ifo[0]
                path_dict[ifo] = try_parse(path).format(i=i, ifo=ifo)
    else:
        path_dict = path
    if abspath:
        path_dict = {k: os.path.abspath(v) for k, v in path_dict.items()}
    return path_dict


def docstring_parameter(*args, **kwargs):
    def dec(obj):
        if obj.__doc__ is not None:
            obj.__doc__ = obj.__doc__.format(*args, **kwargs)
        return obj
    return dec


def string_to_tuple(s):
    result = []
    i = 0
    while i < len(s):
        # Check if character is a digit
        if s[i].isdigit():
            # Check if the previous character was a hyphen
            if i > 0 and s[i-1] == '-':
                # Append as a negative integer
                result.append(-int(s[i]))
            else:
                # Append as a positive integer
                result.append(int(s[i]))
        i += 1
    return tuple(result)


def load_config(config_input):
    if isinstance(config_input, str):
        if os.path.exists(config_input):
            raw_config = ConfigParser()
            raw_config.read(config_input)

            # make subsitutions in case variables are defined in
            # in DEFAULT section
            config = ConfigParser(defaults=None)
            for section in raw_config.sections():
                config.add_section(section)
                for key, value in raw_config.items(section):
                    if key not in raw_config.defaults():
                        config.set(section, key, value)
        else:
            raise FileNotFoundError(config_input)
    elif isinstance(config_input, ConfigParser):
        config = config_input
    elif isinstance(config_input, dict):
        config = ConfigParser()
        for section, options in config_input.items():
            config.add_section(section)
            for key, value in options.items():
                config.set(section, key, str(value))
    else:
        raise ValueError("config_input must be: filename, dict or ConfigParser"
                         f"not {type(config_input)}")
    return config


def load_config_dict(config_input):
    config = load_config(config_input)
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for key, value in config[section].items():
            config_dict[section][key] = try_parse(value)
    return config_dict


class MultiIndexCollection(object):

    def __init__(self, data=None, index=None, reference_mass=None,
                 reference_time=None, info=None) -> None:
        if data is None:
            self.data = []
        elif isinstance(data, dict):
            if index is not None:
                logging.warning("ignoring redundant index ")
            self.data = []
            index = []
            for key, result in data.items():
                self.data.append(result)
                index.append(key if isinstance(key, tuple) else (key,))
        else:
            self.data = [r for r in data]
        self._index = index
        self._reference_mass = reference_mass
        self._reference_time = reference_time
        self.info = info or {}

    @property
    def index(self):
        if self._index is None:
            self._index = [(i,) for i, _ in enumerate(self.data)]
        return self._index

    def __getitem__(self, i):
        return self.data[i]

    def __repr__(self):
        return f"MultiIndexCollection({self.index})"

    def add(self, key, value):
        # Update value if key already exists
        if key in self.keys:
            index = self.index.index(key)
            self.data[index] = value
        else:
            self.index.append(key)
            self.data.append(value)

    def items(self):
        return zip(self.index, self.data)

    def __iter__(self):
        # This allows iteration directly over the key-value pairs
        return iter(self.items())

    def get(self, key):
        return self.data[self.index[key]]

    @property
    def as_dict(self):
        if self._key_size == 1:
            return {k[0]: v for k, v in self.items()}
        else:
            return dict(self.items())

    @property
    def idx(self):
        return self.as_dict

    @property
    def loc(self):
        return self.values

    def __len__(self):
        return len(self.data)

    def __bool__(self):
        return bool(self.data)

    def keys(self):
        return self.index

    def values(self):
        return self.data

    @property
    def _key_size(self):
        if len(self) > 0:
            return len(self.index[0])
        else:
            return 0

    @property
    def reference_mass(self):
        """Reference time relative to which to compute time differences."""
        return self._reference_mass

    @property
    def reference_mass_seconds(self) -> float | None:
        """Reference mass in units of seconds."""
        if self.reference_mass:
            from ..qnms import T_MSUN
            return self.reference_mass * T_MSUN
        else:
            return None

    @property
    def reference_time(self):
        """Reference mass in solar masses to use for time steps in units of
        mass."""
        return self._reference_time

    def set_reference_mass(self, reference_mass):
        if reference_mass is not None:
            reference_mass = float(reference_mass)
        if self._reference_mass is not None:
            logging.warning(
                f"overwriting reference mass ({self._reference_mass})")
        self._reference_mass = reference_mass

    def set_reference_time(self, reference_time):
        if reference_time is not None:
            reference_time = float(reference_time)
        if self._reference_time is not None:
            logging.warning(
                f"overwriting reference time ({self._reference_time})")
        self._reference_time = reference_time

    def reindex(self, new_index):
        if len(new_index) != len(self):
            raise ValueError("New index must have the same length "
                             "as the collection.")
        _new_index = []
        for idx in new_index:
            if not isinstance(idx, tuple):
                idx = (idx,)
            _new_index.append(idx)
        self._index = _new_index


def get_bilby_dict(d):
    """Parse bilby-style data dict string.
    """
    if isinstance(d, str):
        chars_to_remove = "'{}"
        translation_table = str.maketrans('', '', chars_to_remove)
        d = {k.translate(translation_table): v.translate(translation_table)
             for k, v in [i.split(':') for i in d.split(',')]}
    return d
