# HLTRho: Spectral densities from two-point correlation functions

HLTRho is a Python library for operating spectral density reconstruction and
related studies, by using the HLT (Hansen-Lupo-Tantalo) method.

## Authors

Niccolo' Forzano, Alessandro Lupo.

## Installation

Being within the root directory rhos/, where 'pyproject.toml' is located, one can build the package

```bash
python3 -m build
```

This will create a new subdirectory 'dist'. Then, one can install the distribution package

```bash
cd dist/
python3 -m pip install ./HLTRho-0.0.1.tar.gz
```

## Usage

```python
import rhos

# e.g. compute the S_matrix
rhos.core.S_matrix_mp(t_max, alpha)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[GPL](https://choosealicense.com/licenses/gpl-3.0/)
