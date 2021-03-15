# Building documentation

Install dependencies for building documentation by first installing dependencies for the notebooks
(see the [root README.md](../README.md#installation) for instructions). Then run 
```
$ pip install -r requirements.txt -c constraints.txt
```
Build the documentation with
```
$ make html
```
Open `_build/html/index.html` in a browser to view the docs.

# License

[Apache License 2.0](../LICENSE)
