import pytest
import iarray as ia
import json
import requests
import subprocess
import time
import numpy as np
import numpy.testing
import os


ia.ones(shape=[12, 13, 3], urlpath="ones.iarr", mode="w")
ia.arange(shape=[10, 10], urlpath="arange.iarr", dtype=np.int32, mode="w")

dir_path = os.path.dirname(os.path.realpath(__file__))
host = "127.0.0.1"
port = str(ia.HTTP_PORT)

server = subprocess.Popen(["python", "../../scripts/iarr_server.py", host, port, dir_path])
time.sleep(5)


@pytest.fixture(scope="module", autouse=True)
def terminate_server():
    yield
    server.terminate()
    ia.remove_urlpath("ones.iarr")
    ia.remove_urlpath("arange.iarr")


def test_catalog():
    response = requests.get("http://" + host + ":" + port + "/v1/catalog/")
    res = json.loads(response.text)
    assert res == ["ones.iarr", "arange.iarr"]
    assert ia.list_arrays(host, port) == res


@pytest.mark.parametrize(
    "array_id, res", [
        ("ones.iarr", {"shape": [12, 13, 3], "chunks": [8, 8, 2], "blocks": [4, 8, 2], "dtype": "<f8"}),
        ("arange.iarr", {"shape": [10, 10], "chunks": [8, 8], "blocks":[8, 8], "dtype": "<i4"}),
    ]
)
def test_meta(array_id, res):
    url = "http://" + host + ":" + port + "/v1/meta/?array_id=" + array_id
    response = requests.get(url)
    assert json.loads(response.text) == res


@pytest.mark.parametrize(
    "array_id", [
        ("ones.iarr"),
        ("arange.iarr"),
    ]
)
def test_copy(array_id):
    urlpath = "iarr://" + host + ":" + port + "/" + array_id
    a = ia.open(urlpath)

    # Copy
    c = a.data
    d = a.copy()
    numpy.testing.assert_array_equal(c, d.data)


@pytest.mark.parametrize(
    "array_id, expression", [
        ("ones.iarr", "x- cos(0.5)"),
        ("arange.iarr", "x**2"),
    ]
)
def test_server_expr(array_id, expression):
    urlpath = "iarr://" + host + ":" + port + "/" + array_id
    a = ia.open(urlpath)
    # Expr eval
    ia.remove_urlpath("test_expression_zarray.iarr")
    expr = ia.expr_from_string(
        expression,
        {"x": a},
        urlpath="test_expression_zarray.iarr",
    )
    iout = expr.eval()
    npout = ia.iarray2numpy(iout)

    for ufunc in ia.MATH_FUNC_LIST:
        if ufunc in expression:
            idx = expression.find(ufunc)
            # Prevent replacing an ufunc with np.ufunc twice (not terribly solid, but else, test will crash)
            if "np." not in expression[idx - len("np.arc") : idx]:
                expression = expression.replace(ufunc + "(", "np." + ufunc + "(")
    npout2 = eval(expression, {"x": a.data, "np": np})
    np.testing.assert_equal(npout, npout2)

    ia.remove_urlpath(iout.cfg.urlpath)
