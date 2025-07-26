# vfp

Calculate neutron and x-ray reflectivity profiles via generation of volume fraction profiles.

## Installation

```bash
$ pip install vfp
```

## Usage

`vfp` can be used to generate interfacial models.

Lets build a model with the following layers:

$\mathrm{Si}$ | $\mathrm{SiO_{2}}$ | $\mathrm{Surfactant}$ | $\mathrm{D_{2}O}$

where $\mathrm{Si}$ and $\mathrm{D_{2}O}$ are the fronting and backing respectively.

The characteristics of the layers are summarised in the following table.
| | $\mathrm{Thickness}$ / $\mathrm{\mathring A}$ | $\mathrm{Roughness}$ / $\mathrm{\mathring A}$ | $\mathrm{SLD}$ / $\mathrm{\mathring A}^{-2} \times 10^{-6}$ |
| - | - | - | - |
| $\mathrm{Si}$ | $\infty$ | $2$ | $2.07$ |
| $\mathrm{SiO_{2}}$ | $20$ | $4$ | $3.47$ |
| $\mathrm{Surfactant}$ | $30$ | $6$ | $0.21$ |
| $\mathrm{D_{2}O}$ | $\infty$ | \- | $6.37$ |

We can build a simple model with the following:

```python
import matplotlib.pyplot as plt
from vfp import VFP

# fronting thickness = 0 (not $\infty$ here)
# layer 1 thickness = 20
# layer 2 thickness = 30.
thicknesses = (0, 20, 30)

# interfacial width between 0 and 1 = 2,
# between 1 & 2 = 4,
# between 2 & backing = 6
roughnesses = (2, 4, 6)

# slds of Si, SiO2, Surfactant, D2O
slds = (2.07, 3.47, 0.21, 6.37)

# create a VFP object.
vfp = VFP(slds, thicknesses, roughnesses)

# plot SLD, volume fraction proflie & stochastic model of interface.
vfp.plot()
plt.show()
```

The `vfp.refnxVFP` and `vfp.refl1dVFP` objects can be used in [refnx](https://refnx.readthedocs.io/en/latest/) and [refl1d](https://refl1d.readthedocs.io/en/latest/) respectively, e.g with refnx:

```python
from refnx.reflect import ReflectModel, SLD
import matplotlib.pyplot as plt
from vfp import refnxVFP

thicknesses = (0, 20, 30)
roughnesses = (2, 4, 6)
slds = (2.07, 3.47, 0.21, 6.37)

# create a refnxVFP object for refnx
refnx_vfp = refnxVFP(slds, thicknesses, roughnesses)

# wrap the vfp object by the fronting and backing materials
# when defining the structure.
struc = SLD(2.07, name='Si') | refnx_vfp | SLD(6.37, name='D2O')

# now the structure can be used to build a ReflectModel.
model = ReflectModel(struc)

# visualise refnx model via:
plt.plot(*model.structure.sld_profile())
plt.show()

# create your refnx objective & fit / sample however you wish...
```

More in-depth tutorials are found in the docs folder.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`vfp` was created by Alexander Armstrong & Rebecca Welbourn. It is licensed under the terms of the BSD 3-Clause license.

## Credits

`vfp` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
