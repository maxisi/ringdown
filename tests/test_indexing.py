import pytest
import ringdown.indexing
import ringdown.qnms

TUPLE_INDICES = [
    (1, -2, 2, 2, 0),
    (1, -2, 2, 2, 1),
    (1, -2, 3, 2, 0),
]
STR_INDICES_LONG = [
    '1,-2,2,2,0',
    '1,-2,2,2,1',
    '1,-2,3,2,0',
]
STR_INDICES_SHORT = [
    '220',
    '221',
    '320',
]
INT_INDICES = [1, 2, 3]
ALL_INDICES = TUPLE_INDICES + STR_INDICES_LONG +\
              STR_INDICES_SHORT + INT_INDICES


@pytest.mark.parametrize("index", ALL_INDICES)
def test_get_mode_label(index):
    assert isinstance(ringdown.indexing.get_mode_label(index), str)


@pytest.mark.parametrize("index", ALL_INDICES)
def test_get_mode_coordinate(index):
    assert isinstance(ringdown.indexing.get_mode_coordinate(index),
                      (bytes, int))


def _test_index_base(mode):
    assert isinstance(mode, ringdown.indexing.ModeIndex)
    assert isinstance(mode.get_label(), str)
    assert isinstance(mode.get_coordinate(), (bytes, int))
    assert isinstance(mode.is_prograde, bool)
    assert isinstance(mode.as_dict(), dict)


def _test_harmonic(mode, ref_index):
    assert isinstance(mode, ringdown.indexing.HarmonicIndex)
    assert mode.p == ref_index[0]
    assert mode.s == ref_index[1]
    assert mode.l == ref_index[2]
    assert mode.m == ref_index[3]
    assert mode.n == ref_index[4]
    assert isinstance(mode.get_kerr_mode(), ringdown.qnms.KerrMode)
    _test_index_base(mode)


@pytest.mark.parametrize("index", INT_INDICES)
def test_construct_generic(index):
    mode = ringdown.indexing.ModeIndex.construct(index)
    assert isinstance(mode, ringdown.indexing.GenericIndex)
    assert mode.i == index
    _test_index_base(mode)


@pytest.mark.parametrize(["index", "ref_index"],
                         zip(TUPLE_INDICES + STR_INDICES_LONG,
                             TUPLE_INDICES*2))
def test_construct_harmonic(index, ref_index):
    mode = ringdown.indexing.ModeIndex.construct(index)
    _test_harmonic(mode, ref_index)


@pytest.mark.parametrize(["index", "ref_index"],
                         zip(STR_INDICES_LONG + STR_INDICES_SHORT,
                             TUPLE_INDICES))
def test_construct_harmonic_from_string(index, ref_index):
    mode = ringdown.indexing.HarmonicIndex.from_string(index)
    _test_harmonic(mode, ref_index)


@pytest.mark.parametrize("index", TUPLE_INDICES + STR_INDICES_LONG)
def test_bytestrings(index):
    mode = ringdown.indexing.HarmonicIndex.construct(index)
    bs = mode.to_bytestring()
    mode_from_bs = ringdown.indexing.HarmonicIndex.from_bytestring(bs)
    assert mode == mode_from_bs


@pytest.mark.parametrize("indices", [
    TUPLE_INDICES,
    STR_INDICES_LONG,
    STR_INDICES_SHORT,
    INT_INDICES,
])
def test_mode_index_list(indices):
    mil = ringdown.indexing.ModeIndexList(indices)
    assert len(mil.indices) == len(indices)
    assert all(isinstance(m, ringdown.indexing.ModeIndex)
               for m in mil.indices)
