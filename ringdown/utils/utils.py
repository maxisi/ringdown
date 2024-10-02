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


def form_opt(x):
    """Utility to format options in config.
    """
    return np.array2string(np.array(x), separator=', ')


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


def np2(x):
    """Returns the next power of two as big as or larger than x."""
    p = 1
    while p < x:
        p = p << 1
    return p


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
            config = ConfigParser()
            config.read(config_input)
        else:
            raise FileNotFoundError(config_input)
    elif isinstance(config_input, ConfigParser):
        config = config_input
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
            self._index = [(i,) for i in range(len(self.data))]
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

    def __len__(self):
        return len(self.data)

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
        return self._reference_mass

    @property
    def reference_time(self):
        return self._reference_time

    def set_reference_mass(self, reference_mass):
        if reference_mass is not None:
            reference_mass = float(reference_mass)
        self._reference_mass = reference_mass

    def set_reference_time(self, reference_time):
        if reference_time is not None:
            reference_time = float(reference_time)
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
