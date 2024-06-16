import pytest
import ringdown.target
import numpy as np

IFOS = ['H1', 'L1']
REF_SKY_GEO = dict(t0=1126259462.4083147, ra=1.95, dec=-1.27, psi=0.82)
REF_SKY_IFO = dict(t0=1126259462.423, ra=1.95, dec=-1.27, psi=0.82,
                   reference_ifo='H1')
REF_VALUES = {
    'antenna_patterns': {
        'H1': (0.578742411175002, -0.45094782109531206),
        'L1': (-0.5274334329518102, 0.20520960891727422)
    },
    't0': {
        'H1': 1126259462.423,
        'L1': 1126259462.4160156
    }
}
M_REF_MSUN = 70
T_MSUN = 4.92549094764126745135592e-06
M_REF_S = M_REF_MSUN * T_MSUN
T0_REF = REF_SKY_GEO['t0']
DELTA_M = np.arange(5)
T0_LIST = DELTA_M*M_REF_S + T0_REF


def test_t0():
    # NOTE: GPS times are bad for tolerance, we should store nanoseconds
    # separately
    assert all(np.isclose((T0_LIST - T0_REF)/M_REF_S, DELTA_M, rtol=1e-4))


@pytest.mark.parametrize('kws', [REF_SKY_GEO, REF_SKY_IFO, REF_VALUES])
def test_target(kws):
    target = ringdown.target.Target.construct(**kws)
    # check baseclass
    assert isinstance(target, ringdown.target.Target)
    # check subclasses
    assert isinstance(target,
                      (ringdown.target.SkyTarget,
                       ringdown.target.DetectorTarget))
    # check properties
    assert target.is_set
    assert isinstance(target.has_sky, bool)
    assert isinstance(target.has_sky, (bool, None))
    assert isinstance(target.as_dict(), dict)
    assert isinstance(target.sky, tuple)
    # check sky values
    if target.has_sky:
        assert target.t0 == REF_SKY_GEO['t0']
        assert target.ra == REF_SKY_GEO['ra']
        assert target.dec == REF_SKY_GEO['dec']
        assert target.psi == REF_SKY_GEO['psi']
        assert target.sky == (REF_SKY_GEO['ra'],
                              REF_SKY_GEO['dec'],
                              REF_SKY_GEO['psi'])
    # check detector values
    t0s = target.get_detector_times_dict(IFOS)
    aps = target.get_antenna_patterns_dict(IFOS)
    for i in IFOS:
        assert t0s[i] == pytest.approx(REF_VALUES['t0'][i], abs=1e-10)
        assert aps[i] == pytest.approx(REF_VALUES['antenna_patterns'][i],
                                       abs=1e-10)
        assert target.get_detector_time(i) == pytest.approx(t0s[i], abs=1e-10)
        assert target.get_antenna_patterns(i) == pytest.approx(aps[i],
                                                               abs=1e-10)
    # check exceptions
    with pytest.raises(ValueError):
        target.get_detector_time('FAKE')
        target.get_antenna_patterns('FAKE')


def test_simple_target():
    target = ringdown.target.Target.construct(0.)
    assert isinstance(target, ringdown.target.Target)
    assert isinstance(target, ringdown.target.DetectorTarget)
    assert target.is_set
    assert not target.has_sky
    assert target.get_detector_time(None) == 0.
    assert target.get_antenna_patterns(None) == (1., 0.)
    assert target.detector_times == {None: 0.}
    assert target.antenna_patterns == {None: (1., 0.)}
    assert isinstance(target, ringdown.target.Target)


def test_target_collection(rtol=1e-4):
    # manually create a list of targets
    target_list = []
    for t0 in T0_LIST:
        kws = REF_SKY_GEO.copy()
        kws['t0'] = t0
        target_list.append(ringdown.target.Target.construct(**kws))

    # turn into TargetCollection and check it got stored properly
    target_collection = ringdown.target.TargetCollection(target_list)
    assert isinstance(target_collection, ringdown.target.TargetCollection)
    assert len(target_collection) == len(target_list)

    assert isinstance(target_collection.get_antenna_patterns(IFOS[0]), list)
    assert isinstance(target_collection.get_detector_times(IFOS[0]), list)

    # index should be the same as the target's t0 by default
    # and any of the aliases for t0
    assert all(np.equal(target_collection.index, target_collection.t0))
    assert target_collection.index == target_collection.get('t0')
    # since no reference time has been set, delta-t0 should be just t0
    assert all(np.equal(target_collection.t0,
                        target_collection.get('delta-t0')))
    # t0 itself should agree with the initial values we used
    assert all(np.equal(target_collection.t0, T0_LIST))

    # now check individual items
    for i, ((t0, target), ref_target) in enumerate(zip(target_collection,
                                                       target_list)):
        assert t0 == target.t0
        assert target == ref_target
        assert target == target_collection.targets[i]
        assert t0 == ref_target.t0

    # check no reference values
    assert target_collection.reference_time is None
    assert target_collection.reference_mass is None
    assert target_collection.reference_mass_time is None

    # set the reference time
    target_collection.set_reference_time(T0_REF)
    assert target_collection.reference_time == target_list[0].t0

    rel_times = T0_LIST - T0_REF
    assert all(np.isclose(target_collection.get('delta-t0'), rel_times,
                          rtol=rtol))

    # set the reference mass to the default value
    target_collection.set_reference_mass(M_REF_MSUN)
    assert target_collection.reference_mass == M_REF_MSUN
    assert target_collection.reference_mass_time == M_REF_S

    assert all(np.equal(target_collection.t0m,
                        target_collection.get('delta-m')))
    assert all(np.isclose(target_collection.t0m, DELTA_M, rtol=rtol))

    # check exceptions
    with pytest.raises(ValueError):
        ringdown.target.TargetCollection([0.])


if __name__ == "__main__":
    pytest.main()
