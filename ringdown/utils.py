"""Utilities for the ringdown package.
"""

import numpy as np
from ast import literal_eval
import os
from configparser import ConfigParser

def form_opt(x):
    """Utility to format options in config."""
    return np.array2string(np.array(x), separator=', ')

def try_parse(x):
    """Attempt to parse a string as a number, dict, or list."""
    try:
        return float(x)
    except (TypeError,ValueError):
        try:
            return literal_eval(x)
        except (TypeError,ValueError,SyntaxError):
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

def get_path_dict_from_pattern(path, ifos=None):
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
        path_dict = {k: os.path.abspath(v) for k,v in path_dict.items()}
        return path_dict

def docstring_parameter(*args, **kwargs):
    def dec(obj):
        if not obj.__doc__ is None:
            obj.__doc__ = obj.__doc__.format(*args, **kwargs)
        return obj
    return dec

def string_to_tuple(s):
    result = []
    i = 0
    while i < len(s):
        if s[i].isdigit():  # Check if character is a digit
            if i > 0 and s[i-1] == '-':  # Check if the previous character was a hyphen
                result.append(-int(s[i]))  # Append as a negative integer
            else:
                result.append(int(s[i]))  # Append as a positive integer
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