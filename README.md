# matrixor
[![GitHub release][release-image]][release-url]
[![PyPI release][pypi-image]][pypi-url]
[![Build][build-image]][build-url]
[![MIT License][license-image]][license-url]


[release-image]:https://img.shields.io/github/release/akb89/matrixor.svg?style=flat-square
[release-url]:https://github.com/akb89/matrixor/releases/latest
[pypi-image]:https://img.shields.io/pypi/v/matrixor.svg?style=flat-square
[pypi-url]:https://pypi.org/project/matrixor/
[build-image]:https://img.shields.io/github/workflow/status/akb89/matrixor/CI?style=flat-square
[build-url]:https://github.com/akb89/matrixor/actions?query=workflow%3ACI
[license-image]:http://img.shields.io/badge/license-MIT-000000.svg?style=flat-square
[license-url]:LICENSE.txt

Matrix transformations in Python. Implements algorithm 2.4 (AO+Scaling) of the paper:

```tex
@article{devetal2018,
    title={{Absolute Orientation for Word Embedding Alignment}},
    author={Sunipa Dev and Safia Hassan and Jeff M. Phillips},
    journal={CoRR},
    year={2018},
    volume={abs/1806.01330}
}
```

## Install
```shell
pip install matrixor
```

or, after a git clone:
```shell
python3 setup.py install
```

## Run

### Align two models with Absolute Orientation + Scaling and return RMSE
To align two embeddings, run:
```shell
matrixor align \
  --matrix-1 /abs/path/to/xxx.vec.npy  \
  --matrix-2 /abs/path/to/yyy.vec.npy \
```
pass in the `--average` to return the averaged RMSE by aligning A and B then B and A (alignment may not be symmetric due to floating point average)
