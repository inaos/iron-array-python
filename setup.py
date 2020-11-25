import os
from sys import platform
from skbuild import setup
import codecs
import os.path


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


BUILD_WHEELS = True if "BUILD_WHEELS" in os.environ else False
if not BUILD_WHEELS:
    if os.path.exists("BUILD_WHEELS"):
        BUILD_WHEELS = True

DESCRIPTION = "The Math Array Accelerator for Python"


with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

# Libraries to copy as 'data' in package
# Copying just 'libarray' seems good enough
if platform == "linux" or platform == "linux2":
    copy_libs = ["libiarray.so", "libsvml.so", "libintlc.so.5"]
elif platform == "darwin":
    copy_libs = ["libiarray.dylib"]
elif platform == "win32":
    copy_libs = ["iarray.dll", "svml_dispmd.dll"]
else:
    copy_libs = []

doc_deps = [
    "sphinx >= 1.5",
    "sphinx_rtd_theme",
    "numpydoc",
]
examples_deps = [
    "matplotlib",
    "numexpr",
    "numba",
]

if BUILD_WHEELS:
    print("BUILD_WHEELS mode is ON!")
    install_requires = open("requirements-runtime.txt").read().split()
    package_info = dict(
        package_dir={"iarray": "iarray"},
        packages=["iarray", "iarray.py2llvm", "iarray.tests"],
        package_data={"iarray": copy_libs},
        install_requires=install_requires,
    )
else:
    # For some reason this is necessary for inplace compilation
    # One can avoid using this if we nuke _skbuild/ next to iarray/
    package_info = dict(
        package_dir={"": "."},
    )

setup(
    name="iarray",
    version=get_version("iarray/__init__.py"),
    description=DESCRIPTION,
    # long_description=LONG_DESCRIPTION,
    python_requires=">=3.7",
    extras_require={"doc": doc_deps, "examples": examples_deps},
    **package_info,
)
