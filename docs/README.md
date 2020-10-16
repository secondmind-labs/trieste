# Building documentation

Install dependencies for building documentation by first installing dependencies for the notebooks
(see the [root REAMDE.md](..README.md) for instructions). Then run 
```
$ pip install -r ../deps/docs/requirements.txt -c ../deps/docs/constraints.txt
```
Build the documentation with
```
$ make html
```
Open `_build/html/index.html` in a browser to view the docs.

# License

[Apache License 2.0](../LICENSE)
