# lsdensities: Spectral densities from two-point correlation functions

lsdensities is a Python library for operating spectral density reconstruction and
related studies, by using the HLT (Hansen-Lupo-Tantalo) method or Gaussian processes.

## Authors

Niccolò Forzano, Alessandro Lupo.

## Installation

One can download, build and install the package

```bash
pip install https://github.com/LupoA/rhos
```

## Usage

```python
import lsdensities

# e.g. compute the S matrix
lsdensities.core.Smatrix_mp(t_max, alpha)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[GPL](https://choosealicense.com/licenses/gpl-3.0/)
