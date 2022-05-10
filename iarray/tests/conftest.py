import iarray as ia
import pytest


@pytest.fixture(autouse=True)
def reset_config():
    ia.reset_config_defaults()


@pytest.fixture(scope='session', autouse=True)
def cleanup(request):
    return request.addfinalizer(ia.udf_registry.clear)
