# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Multiple Geometry / Probe Example

# %% [markdown]
# If we measure the same sample in multiple ways (e.g. x-ray and neutron) we want to be able to co-refine the data for a single sample across the multiple measurements. This example fits the same sample with 2 different probes. In this case, it is using a solid-air and an air-solid measurement.

# %%
import numpy as np
import pathlib
import os
import matplotlib.pyplot as plt
import periodictable

from refnx.analysis import GlobalObjective, Parameter, Objective, CurveFitter
from refnx.reflect import ReflectModel, SLD
from refnx.dataset import ReflectDataset

import vfp

# %% [markdown]
# ### Sample Description:
#
# The sample is a thin film of Iridium on a silicon wafer which has been measured in the two geometries, i.e.
#
#     Si | SiOx | Ir | Air
#
#     Air | Ir | SiOx | Si 

# %%
# Define the SLD / materials used in the model:

Si = SLD(2.07, name='Si')
SiO2 = SLD(1.96, name='SiO2')
Ir = SLD(7.0, name='Ir')
Air = SLD(0.0, name='Air')

# %%
# Create parameters for the parameters we will vary (i.e. the thickness, roughness and volume fractions:

sio2_thickness = Parameter(38, name='sio2_thickness', vary=True, bounds=(10, 50))
Ir_thickness = Parameter(760, name='Ir_thickness', vary=True, bounds=(600, 850))

si_sio2_roughness = Parameter(1, name='si_sio2_roughness', vary=True, bounds=(1, 5))
sio2_Ir_roughness = Parameter(2, name='sio2_Ir_roughness', vary=True, bounds=(1, 6))
Ir_air_roughness = Parameter(25, name='Ir_air_roughness', vary=True, bounds=(6, 35))

# %%
# Assemble lists in the structure order for our thickness, roughness and slds

thicknesses = (0, sio2_thickness, Ir_thickness)
roughnesses = (si_sio2_roughness, sio2_Ir_roughness, Ir_air_roughness)

list_slds = [Si.real,
               SiO2.real,
               Ir.real,
               Air.real]


# %%
# Assemble these into the VFP objects for each contrast:

Si_vfp = vfp.refnxVFP(nslds=list_slds,
                     thicknesses=thicknesses,
                     roughnesses=roughnesses,
                     orientation="front")

Air_vfp = vfp.refnxVFP(nslds=list_slds,
                     thicknesses=thicknesses,
                     roughnesses=roughnesses,
                     orientation="back")

# %%
# Define our structures

Si_vfp_struc = Si | Si_vfp | Air
Air_vfp_struc = Air | Air_vfp | Si

# Allow background and intensity to vary
Si_int = Parameter(1, name='Si_int', vary=True, bounds=(0.9, 1.1))
Si_bkg = Parameter(1e-6, name='Si_bkg', vary=True, bounds=(1e-9, 1e-6))
Air_int = Parameter(1, name='Air_int', vary=True, bounds=(0.9, 1.1))
Air_bkg = Parameter(1e-6, name='Air_bkg', vary=True, bounds=(1e-9, 1e-6))

# %%
# Define our models

Si_model = ReflectModel(structure=Si_vfp_struc, scale=Si_int, bkg=Si_bkg)
Air_model = ReflectModel(structure=Air_vfp_struc, scale=Air_int, bkg=Air_bkg)

# %%
# Load in the data -- NEED TO UPDATE THIS!!

dataset_dir = pathlib.Path(vfp.__path__[0]).parents[1] / 'docs' / 'test_data'

Si_load = np.loadtxt(dataset_dir / "Multimodel_SiAir.txt")
Air_load = np.loadtxt(dataset_dir / "Multimodel_AirSi.txt")


Si_data = ReflectDataset((Si_load[:, 0], Si_load[:, 1], Si_load[:, 2], Si_load[:, 3]))
Air_data = ReflectDataset((Air_load[:, 0], Air_load[:, 1], Air_load[:, 2], Air_load[:, 3]))

# %%
# Link the models and data into our Objectives

Si_obj = Objective(model=Si_model, data=Si_data)
Air_obj = Objective(model=Air_model, data=Air_data)
global_objective = GlobalObjective([Si_obj, Air_obj])

# %%
# Show the plot for the pre-fit state:
fig,ax=global_objective.plot()
plt.title("Pre-fit Model & Data")
plt.xscale('log')
plt.yscale('log')

# %%
from pathos.pools import ProcessPool

pool = ProcessPool(nodes=4)
fitter = CurveFitter(objective=global_objective)
fitter.fit(method='differential_evolution', workers=pool.map)

# %%
# Show the plot for the pre-fit state:
fig,ax=global_objective.plot()
plt.title("Post-fit Model & Data")
plt.xscale('log')
plt.yscale('log')

# %%
print(global_objective.varying_parameters())

# %%
## Note: this might need updating with new plotting features.
Si_vfp.plot()
plt.show()
Air_vfp.plot()
plt.show()

# %%
fig, ax = plt.subplots()
for vfp_obj, label in [(Si_vfp, "Si"), (Air_vfp, "Air")]:
    z, all_slds = vfp_obj.z_and_sld()
    ax.plot(z + vfp_obj.sld_offset(), all_slds[:, 0], label=label)  # sld_n
ax.set_ylabel("SLD / Å⁻² ×10⁻⁶")
ax.set_xlabel("Distance / Å")
ax.legend(frameon=False)
plt.title("SLD Profiles After Fit")
plt.show()

# %%
fig, ax = plt.subplots()
for vfp_obj, label, offset, rev in [(Si_vfp, "Si", 0, 1), (Air_vfp, "Air", -798, -1)]:
    z, all_slds = vfp_obj.z_and_sld()
    ax.plot((z + offset)*rev, all_slds[:, 0], label=label)  # sld_n
ax.set_ylabel("SLD / Å⁻² ×10⁻⁶")
ax.set_xlabel("Distance / Å")
ax.legend(frameon=False)
plt.title("SLD Profiles After Fit")
plt.show()

# %%
