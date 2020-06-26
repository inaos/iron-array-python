.PHONY: build install upgrade mrproper

build:
	./venv37/bin/python setup.py build_ext -j4 --build-type=Debug -- -DDISABLE_LLVM_CONFIG=OFF

install:
	git submodule update --init --recursive
	python3.7 -m venv venv37
	./venv37/bin/pip install -U pip
	./venv37/bin/pip install -r requirements.txt
	./venv37/bin/pip install Cython numba hypothesis

upgrade:
	git submodule update --remote --recursive

mrproper:
	git clean -fXd
