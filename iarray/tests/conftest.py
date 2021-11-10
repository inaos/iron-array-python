import iarray as ia
import pytest


@pytest.fixture(autouse=True)
def reset_config():
    ia.reset_config_defaults()
