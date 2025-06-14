__all__ = ["ParameterLabel", "get_parameter_label_map", "get_latex_from_key"]

from . import indexing


class ParameterLabel(object):
    _PARAMETER_KEY_MAP = {
        "m": "$M / M_\\odot$",
        "chi": "$\\chi$",
        "f": "$f_{{{mode}}} / \\mathrm{{Hz}}$",
        "g": "$\\gamma_{{{mode}}} / \\mathrm{{Hz}}$",
        "df": "$\\delta f_{{{mode}}}$",
        "dg": "$\\delta \\gamma_{{{mode}}}$",
        "a": "$A_{{{mode}}}$",
        "phi": "$\\phi_{{{mode}}}$",
        "theta": "$\\theta_{{{mode}}}$",
        "ellip": "$\\epsilon_{{{mode}}}$",
        "h_det": "$h(t) [\\mathrm{{{ifo}}}]$",
        "h_det_mode": "$h_{{{mode}}}(t) [\\mathrm{{{ifo}}}]$",
    }

    def __init__(self, parameter):
        self.parameter = parameter.lower()
        if self.parameter not in self._PARAMETER_KEY_MAP:
            raise ValueError(f"Parameter {parameter} not recognized.")

    def __str__(self):
        return self.parameter

    def __repr__(self):
        return f"ParameterLabel('{self.parameter}')"

    @property
    def is_mode_specific(self):
        label = self._PARAMETER_KEY_MAP[self.parameter]
        return "{{{mode}}}" in label

    @property
    def is_strain(self):
        return self.parameter.startswith("h_det")

    def get_latex(self, mode=None, ifo=None, **kws):
        label = self._PARAMETER_KEY_MAP[self.parameter]
        subst = {}
        if mode is not None:
            mode_index = indexing.get_mode_label(mode, **kws)
            subst["mode"] = mode_index
        elif self.is_mode_specific:
            label = label.replace("_{{{mode}}}", "")
        if ifo is not None:
            subst["ifo"] = ifo
        else:
            label = label.replace(" [\\mathrm{{{ifo}}}]", "")
        return label.format(**subst)

    def get_key(self, mode=None, ifo=None, **kws):
        key = self.parameter
        if mode is not None:
            mode_index = indexing.get_mode_label(mode, **kws)
            if key == "h_det_mode":
                key = key.replace("mode", mode_index)
            elif self.is_mode_specific:
                key = f"{key}_{mode_index}"
        if ifo is not None:
            key = key.replace("det", ifo)
        return key

    def get_label(self, latex=False, **kws):
        if latex:
            return self.get_latex(**kws)
        else:
            return self.get_key(**kws)


def get_parameter_label_map(pars=None, modes=None, ifos=None, **kws):
    label_dict = {}
    pars = pars or ParameterLabel._PARAMETER_KEY_MAP.keys()
    if modes is None:
        modes = [None]
    if ifos is None:
        ifos = [None]
    for k in pars:
        p = ParameterLabel(k)
        for i in ifos:
            for m in modes:
                label_dict[p.get_key(mode=m, ifo=i, **kws)] = p.get_latex(
                    mode=m, ifo=i, **kws
                )
    return label_dict


def get_latex_from_key(key):
    param_mode = key.split('_')
    if len(param_mode) == 2:
        param, mode = param_mode
        return ParameterLabel(param).get_latex(mode=mode)
    else:
        return ParameterLabel(key).get_latex()
