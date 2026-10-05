import pytest

from wefnexus.data import example_basin
from wefnexus.models import Scenario


@pytest.fixture
def basin():
    """Fresh copy of the stylised Azura River basin."""
    return example_basin()


@pytest.fixture
def baseline():
    return Scenario(name="baseline", years=5)
