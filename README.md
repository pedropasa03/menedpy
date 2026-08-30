# menedpy

This repository contains a Python library suited for solving PDE's numerically

## How Python modules work

Python can only import `menedpy` if it can find the package on its import path.
That usually happens in one of two ways:

1. You install the package into your environment.
2. You run a script from a folder that already contains the package.

Because this project uses a `src/` layout, the code lives in `src/menedpy/`.
That is a good practice, but it means a plain script like `examples/h.py` will not
see the package automatically unless the package is installed first.

## Install

From the project root:

```bash
python -m pip install -e . --no-build-isolation
```

That installs the package in editable mode, so changes in `src/menedpy/` are
immediately visible without reinstalling.

## Example

After installing, this works from anywhere:

```python
from menedpy.newton import newton_system
```
