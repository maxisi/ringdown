import warnings

import pytest


@pytest.fixture(scope="session", autouse=True)
def _pandas_futurewarning_errors():
    # Copy-on-Write is always on in pandas 3; still fail on pandas FutureWarnings.
    warnings.filterwarnings("error", category=FutureWarning, module=r"pandas(\.|$)")
