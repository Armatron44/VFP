# vfp

Calculate neutron and x-ray reflectivity profiles via generation of volume fraction profiles.

## Installation

```bash
$ pip install vfp
```

## Usage

`vfp` can be used to generate interfacial models. 
A simple model of an interface can be generated with the following:

```python
import matplotlib.pyplot as plt
import vfp

# fronting thickness = 0, layer 1 thickness = 20, layer 2 thickness = 30.
thicknesses = (0, 20, 30)
# interfacial width between 0 and 1 = 2, between 1 & 2 = 4, between 2 & backing = 6
roughnesses = (2, 4, 6)
# slds of fronting = Si, layer 1 = SiO2, layer 2 = GMO, backing = D2O
slds = (2.07, 3.47, 0.21, 6.37)
# create a refnxVFP object.
vfp = vfp.refnxVFP(slds, lot, lor)
# plot SLD, volume fraction proflie & stochastic model of interface.
vfp.plot()
plt.show()
```

The `vfp.refnxVFP` and `vfp.refl1dVFP` objects can be used in `refnx` and `refl1d` respectively, e.g:

```python
from refnx.analysis import GlobalObjective, Parameter, Objective, CurveFitter
from refnx.reflect import ReflectModel, SLD
from refnx.dataset import ReflectDataset
import matplotlib.pyplot as plt
import vfp

# fronting thickness = 0, layer 1 thickness = 20, layer 2 thickness = 30.
thicknesses = (0, 20, 30)
# interfacial width between 0 and 1 = 2, between 1 & 2 = 4, between 2 & backing = 6
roughnesses = (2, 4, 6)
# slds of fronting = Si, layer 1 = SiO2, layer 2 = GMO, backing = D2O
slds = (2.07, 3.47, 0.21, 6.37)
# create a refnxVFP object.
vfp = vfp.refnxVFP(slds, lot, lor)

# wrap the vfp object by the fronting and backing materials when defining the structure.
struc = SLD(2.07, name='Si') | vfp | SLD(6.37, name='D2O')

# now the structure can be used to build a ReflectModel as normal.
model = ReflectModel(struc)

# visualise refnx model via:
plt.plot(*model.structure.sld_profile())
plt.show()
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`vfp` was created by Alexander Armstrong & Rebecca Welbourn. It is licensed under the terms of the BSD 3-Clause license.

## Credits

`vfp` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
