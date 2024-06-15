import pytest
import ringdown.qnms
import ringdown.result

T_MSUN = 4.9254909476412675e-06
M_REF = 70
C_REF = 0.7
FTAU_REF = {
    (1, -2, 2, 2, 0): (245.85210273356427, 0.004267509656122605),
    (1, -2, 2, 2, 1): (240.57155728864586, 0.001411671895902423),
    (1, -2, 3, 2, 0): (350.440511991165, 0.004095329813033091)
}


def test_T_MSUN():
    assert ringdown.qnms.T_MSUN == T_MSUN


@pytest.mark.parametrize("index, f_tau_ref", FTAU_REF.items())
def test_get_ftau(index, f_tau_ref):
    (p, s, l, m, n) = index
    ftau = ringdown.qnms.get_ftau(M_REF, C_REF, n=n, l=l, m=p*m)
    assert ftau == pytest.approx(f_tau_ref, rel=1e-12)


class TestKerrMode:

    def setup_method(self, method):
        self.mass = M_REF
        self.spin = C_REF
        self.kerr_modes = {m: ringdown.qnms.KerrMode(*m)
                           for m in FTAU_REF.keys()}

    def test_ftau(self):
        for (p, s, l, m, n), mode in self.kerr_modes.items():
            ref = ringdown.qnms.get_ftau(self.mass, self.spin, n=n, l=l, m=p*m)
            assert mode.ftau(self.spin, self.mass) == ref

    def test_fgamma(self):
        for (p, s, l, m, n), mode in self.kerr_modes.items():
            f_ref, t_ref = ringdown.qnms.get_ftau(self.mass, self.spin, n=n,
                                                  l=l, m=p*m)
            f, g = mode.fgamma(self.spin, self.mass)
            assert (f, g) == pytest.approx((f_ref, 1/t_ref), abs=1e-12)

    @pytest.mark.slow
    def test_ftau_approx(self):
        for (p, s, l, m, n), mode in self.kerr_modes.items():
            ref = ringdown.qnms.get_ftau(self.mass, self.spin, n=n, l=l, m=p*m)
            ftau = mode.ftau(self.spin, self.mass, approx=True)
            assert ftau == pytest.approx(ref, rel=1E-2)


def test_get_parameter_label_map():
    ringdown.qnms.get_parameter_label_map(
        pars=ringdown.result._DATAFRAME_PARAMETERS,
        modes=FTAU_REF.keys())


@pytest.mark.parametrize("parameter", ringdown.result._DATAFRAME_PARAMETERS)
def test_parameter_label(parameter):
    p = ringdown.qnms.ParameterLabel(parameter)
    assert isinstance(p.get_label(latex=True), str)
    assert isinstance(p.get_label(latex=False), str)
    assert isinstance(p.get_key(), str)


if __name__ == "__main__":
    pytest.main()
