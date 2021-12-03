# Guidelines for building the documentation

When some changes are made in the tutorials or benchmarks notebooks (the ones inside the `doc/source/bench/` or `doc/source/tutorials/` directories), a specific order has to be followed in order to properly run the notebooks:

1. Run the tutorials
2. Run the benchmarks

Then, the documentation will be generated as usual with:

```bash
cd doc
PYTHONPATH=.. make html 
```
