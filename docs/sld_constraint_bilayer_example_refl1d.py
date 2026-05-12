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
# # Constraining SLD: Bilayer (Refl1D)
#
# In some cases, we may want to constrain sld parameters via the integrals of layers in the `vfp`. This is especially useful when attempting to constrain the amount of material in one layer with the amount of material in a second layer.
#
# In this example, we show how to use the `sld_constraint` parameter that can be supplied to `vfp.VFP`, `vfp.refnxVFP` and `vfp.refl1dVFP` to achieve such a constraint.

# %%
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import periodictable
from bumps.parameter import Parameter
from bumps.fitproblem import FitProblem
from refl1d.experiment import Experiment
from refl1d.sample.material import SLD
from refl1d.probe.data_loaders.load4 import load4

# Local
import vfp


# %% [markdown]
# ## A contrived example
#
# We'll build a model where we have a substrate (silicon, with a silicon oxide layer) submerged in a water, with an adsorbed bilayer (e.g. lipid or surfactant). We *could* model the adsorbed material as layers containing the average of head and tail portions, but we can build a more informed model by considering the head and tail portions separately but with constraints to conserve the number of molecules across the head-tail pairing. An overview of the structure looks like this:
#
#     Si | SiOx | Inner head | Inner tail | Outer tail | Outer head | Water
#
# We've measured three contrasts of
# - $\mathrm{D}_{2}\mathrm{O}$ + h-surfactant
# - $\mathrm{H}_{2}\mathrm{O}$ + h-surfactant 
# - $\mathrm{D}_{2}\mathrm{O}$ + tail-deuterated surfactant
#
# ### Head layer
# We'll model the inner layer as containing surfactant head groups and some water, so the SLD of the inner layer is
# $$\mathrm{SLD} = \mathrm{SLD}_{\mathrm{w}} * \phi_{\mathrm{w}} + \mathrm{SLD}_{\mathrm{h}} * \phi_{\mathrm{h}},$$
# where $\mathrm{w}$ and $\mathrm{h}$ denote water and head group in the layer, and $\phi$ denotes their respective volume fractions in the layer. The sum of the volume fractions must add up to one, i.e $\sum_{j}{\phi_{j}} = 1$.
#
# These parameters can all be specified relatively simply without any additional constraints outside of the flexibility provided by `refnx` or `refl1d`.
#
# ### Tail layer
# An equivalent expression can be written for the tail layer
# $$\mathrm{SLD} = \mathrm{SLD}_{\mathrm{w}}*\phi_{\mathrm{w}} + \mathrm{SLD}_{\mathrm{t}}*\phi_{\mathrm{t}},$$
# where $\mathrm{t}$ denotes the surfactant tail group in the layer. Again $\sum_{j}*{\phi_{j}}=1$.
#
# We assume that the water component described is a coverage parameter and is common across a head-tail pair. There may also be additional water molecules associated with the head (or tail) groups but this should be included in the head part ($\mathrm{SLD}_{\mathrm{h}}$).
#
# ## Conserving the number of surfactant molecules
#
# The surface excess (number of units per unit area), $\Gamma$, of head groups and tail groups in a given layer must be equal to ensure a monolayer, i.e , $\Gamma_\mathrm{t, in} = \Gamma_\mathrm{h, in}$. The surface excess of a material $m$ in layer $j$ can be parameterised as 
#
# $$ \Gamma_{m} = \phi_{m, j}\frac{\mathrm{SLD}_{m}}{b_{m}N_{\mathrm{A}}}\int{\Phi_{j}\left(z\right) \mathrm{d}z},$$
#
# where $b_{m}$ is the scattering length of the material, $N_{\mathrm{A}}$ is Avogadro's constant and $\Phi_{j}\left(z\right)$ is the volume fraction profile of the $j\text{th}$ layer.
#
# Therefore we can constrain $\phi_{\mathrm{t}}$ to be a function of $\phi_{\mathrm{h}}$ over the Head and Tail paired layers
#
# $$
# \begin{aligned}
# \phi_{\mathrm{t}} \frac{\mathrm{SLD}_{\mathrm{t}}}{b_{\mathrm{t}}N_{\mathrm{A}}}\int{\Phi_{\mathrm{Tail}}\left(z\right) \mathrm{d}z} &= \phi_{\mathrm{h}}\frac{\mathrm{SLD}_{\mathrm{h}}}{b_{\mathrm{h}}N_{\mathrm{A}}}\int{\Phi_{\mathrm{Head}}\left(z\right) \mathrm{d}z} \\
# \phi_{\mathrm{t}} &= \phi_{\mathrm{h}}\frac{\mathrm{SLD}_{\mathrm{h}}b_{\mathrm{t}}}{b_{\mathrm{h}}\mathrm{SLD}_{\mathrm{t}}}\frac{\int{\Phi_{\mathrm{Head}}\left(z\right) \mathrm{d}z}}{\int{\Phi_{\mathrm{Tail}}\left(z\right) \mathrm{d}z}}
# \end{aligned}
# $$
#
# This expression is separately applied to the inner and outer leaflets.
#
# Let's set up some refnx SLDs objects.

# %% [markdown]
# ## Setup the constraint as a class:

# %%
class SLDConstraint:
    """
    Class with two methods that are used by the VFP.
    `layer_choices` returns the indices of the layers that we'll constrain together.
    '__call__' Uses the  a function that   
    """
    def __init__(self, required_pars: dict[float]):
        self.required_pars = required_pars
        
    def layer_choices(self) -> list[int]:
        return [2, 3, 4, 5]
        
    def __call__(self, layer_integrals: list[float]) -> tuple[list[int], list[float]]:
        lay2_int, lay3_int, lay4_int, lay5_int = layer_integrals
        
        surf_inner_head_vf, surf_inner_head_sld, surf_inner_tail_sld, b_surf_inner_head, b_surf_inner_tail, surf_outer_head_vf, surf_outer_head_sld, surf_outer_tail_sld, b_surf_outer_head, b_surf_outer_tail, water_sld = map(
            self.required_pars.get,
            ['surf_inner_head_vf', 
             'surf_inner_head_sld',
             'surf_inner_tail_sld',
             'b_surf_inner_head',
             'b_surf_inner_tail',
             'surf_outer_head_vf',
             'surf_outer_head_sld',
             'surf_outer_tail_sld',
             'b_surf_outer_head',
             'b_surf_outer_tail', 
             'water_sld'
             ]
        )
        
        # using the integrals, constrain the volume fraction of the tail
        surf_tail_vf_inner = (surf_inner_head_vf
                    * ((surf_inner_head_sld * b_surf_inner_tail) / (b_surf_inner_head * surf_inner_tail_sld)) 
                    * (lay2_int / lay3_int)
        )
        
        surf_tail_vf_outer = (surf_outer_head_vf
                    * ((surf_outer_head_sld * b_surf_outer_tail) / (b_surf_outer_head * surf_outer_tail_sld)) 
                    * (lay5_int / lay4_int)
        )
        
        inner_water_vf = 1 - (surf_tail_vf_inner)
        outer_water_vf = 1 - (surf_tail_vf_outer)
        
        # now we have outer_water_vf, we can calculate the true sld of the outer layer:
        layer_idx = [3, 4] # outer layer sld index
        altered_sld = [
            (inner_water_vf*water_sld + surf_tail_vf_inner*surf_inner_tail_sld),
            (outer_water_vf*water_sld + surf_tail_vf_outer*surf_outer_tail_sld)
        ]
        
        return layer_idx, altered_sld


# %% [markdown]
# ## Breakdown of the `__call__` method above
#
# The call method of `SLDConstraint` takes a list of floats, which are the integrals of the requested layers from `SLDConstraint.layer_choices`. These are passed to `SLDConstraint` by `vfp`. 
#
# Using these integral values with the appropriate parameters (these will be passed to `SLDConstraint` when we initialise it), we calculate the volume fraction of the tail group for each leaflet. As we now know the head group volume fraction, and the water volume fraction can be determined. This means we can calcluate the sld of the tail layer with this constraint.
#
# To update (possibly multiple) sld values in `vfp`, we return the layer indices and sld values in separate lists, and `vfp` will update the sld values for the layers specified by their indices.  

# %% [markdown]
# ## Load test data:

# %%
dataset_dir = pathlib.Path(vfp.__path__[0]).parents[1] / 'docs' / 'test_data'

d2o_probe = load4(filename=str(dataset_dir/"Bilayer_constraint_testdata_d2o.txt"), back_reflectivity=True, background=1e-6)
h2o_probe = load4(filename=str(dataset_dir/"Bilayer_constraint_testdata_h2o.txt"), back_reflectivity=True, background=1e-6)
d2o_dtails_probe = load4(filename=str(dataset_dir/"Bilayer_constraint_testdata_d2o_dtails.txt"), back_reflectivity=True, background=1e-6)

# Allow background and scaling-factors / intensities to vary

d2o_probe.background = Parameter(1e-6, name='d2o_background', bounds=(1e-7, 1e-4))
d2o_probe.intensity  = Parameter(1.0, name='d2o_intensity',  bounds=(0.9, 1.1))

h2o_probe.background = Parameter(1e-6, name='h2o_background', bounds=(1e-7, 1e-4))
h2o_probe.intensity  = Parameter(1.0, name='h2o_intensity',  bounds=(0.9, 1.1))

d2o_dtails_probe.background = Parameter(1e-6, name='d2o_dtails_background', bounds=(1e-7, 1e-4))
d2o_dtails_probe.intensity  = Parameter(1.0, name='d2o_dtails_intensity',  bounds=(0.9, 1.1))


# %% [markdown]
# ## Setup a model to test the constraints:

# %%
# Describe a dummy surfactant molecule:

number_density = (853 / ((periodictable.C.mass * 16 + periodictable.H.mass* 32 + periodictable.O.mass*2)*1660.854))
print(f'number density = {number_density:.3e} Å^{-3}')

b_head = periodictable.C.neutron.b_c + 2*periodictable.O.neutron.b_c + periodictable.H.neutron.b_c

b_htail = 15*periodictable.C.neutron.b_c + 31*periodictable.H.neutron.b_c
b_dtail = 15*periodictable.C.neutron.b_c + 31*periodictable.D.neutron.b_c

b_full_d = periodictable.C.neutron.b_c * 16 + periodictable.H.neutron.b_c + 31*periodictable.D.neutron.b_c + periodictable.O.neutron.b_c*2
b_full_h = periodictable.C.neutron.b_c * 16 + periodictable.H.neutron.b_c + 31*periodictable.H.neutron.b_c + periodictable.O.neutron.b_c*2

print(f'b head = {b_head*1E-5:.3e} Å')
print(f'b h-tail = {b_htail*1E-5:.3e} Å')
print(f'b d-tail = {b_dtail*1E-5:.3e} Å')
print(f'b full d = {b_full_d*1E-5:.3e} Å = b head + b dtail = {1E-5 * (b_head + b_dtail):.3e} Å')
print(f'b full h = {b_full_h*1E-5:.3e} Å = b head + b htail = {1E-5 * (b_head + b_htail):.3e} Å')


print(f'sld of head = {number_density*b_head*1E-5*1E6:.3e} Å^{-2} x 10^-6')

print(f'sld of htail = {number_density*b_htail*1E-5*1E6:.3e} Å^{-2} x 10^-6')
print(f'sld of dtail = {number_density*b_dtail*1E-5*1E6:.3e} Å^{-2} x 10^-6')

print(f'sld of full h = {number_density*b_full_h*1E-5*1E6:.3e} Å^{-2} x 10^-6')
print(f'sld of full d = {number_density*b_full_d*1E-5*1E6:.3e} Å^{-2} x 10^-6')

# %%
# Define the SLD / materials used in the model:

Si_mat = SLD(rho=2.07, name='Si')
SiO2 = SLD(rho=3.47, name='SiO2')
surf_head_inner = SLD(rho=0.291, name='surf_head_inner')
surf_h_tail = SLD(rho=-0.326, name='surf_h-tail')
surf_d_tail = SLD(rho=6.137, name='surf_h-tail')
surf_head_outer = SLD(rho=0.291, name='surf_head_outer')
d2o = SLD(rho=6.37, name='d2o')
h2o = SLD(rho=-0.54, name='h2o')

# %%
# Create parameters for the parameters we will vary (i.e. the thickness, roughness and volume fractions:

sio2_thickness = Parameter(20, name='sio2_thickness', vary=True, bounds=(10, 30))
inner_head_thickness = Parameter(5, name='inner_head_thickness', vary=True, bounds=(1, 7))
outer_head_thickness = Parameter(5, name='outer_head_thickness', vary=True, bounds=(1, 7))
inner_tail_thickness = Parameter(10, name='inner_tail_thickness', vary=True, bounds=(8, 20))
outer_tail_thickness = Parameter(10, name='outer_tail_thickness', vary=True, bounds=(8, 20))

si_sio2_roughness = Parameter(1, name='si_sio2_roughness', vary=True, bounds=(1, 5))
sio2_inner_roughness = Parameter(1, name='sio2_inner_roughness', vary=True, bounds=(1, 6))
inner_outer_roughness = Parameter(1, name='inner_outer_roughness', vary=True, bounds=(1, 5))
outer_solv_roughness = Parameter(1, name='outer_solv_roughness', vary=True, bounds=(1, 6))

surf_inner_vf = Parameter(0.5, name='surf_inner_vf', vary=True, bounds=(0, 1))
surf_outer_vf = Parameter(0.5, name='surf_outer_vf', vary=True, bounds=(0, 1))

# Specify the b values for the head and tail groups. These can be parameters or calculated.
b_heads_inner=b_head
b_heads_outer=b_head
b_htails=b_htail
b_dtails=b_dtail

# %%
# Assemble lists in the structure order for our thickness, roughness and slds

thicknesses = (0, sio2_thickness, inner_head_thickness, inner_tail_thickness, outer_tail_thickness, outer_head_thickness)
roughnesses = (si_sio2_roughness, sio2_inner_roughness, inner_outer_roughness,inner_outer_roughness,inner_outer_roughness, outer_solv_roughness)

d2o_slds = [Si_mat.rho,
               SiO2.rho,
               surf_inner_vf * surf_head_inner.rho + (1-surf_inner_vf) * d2o.rho,
               1, # this will be updated by the vfp.
               1, # this will be updated by the vfp.
               surf_outer_vf * surf_head_outer.rho + (1-surf_outer_vf) * d2o.rho,
               d2o.rho]

h2o_slds = [Si_mat.rho,
               SiO2.rho,
               surf_inner_vf * surf_head_inner.rho + (1-surf_inner_vf) * h2o.rho,
               1, # this will be updated by the vfp.
               1, # this will be updated by the vfp.
               surf_outer_vf * surf_head_outer.rho + (1-surf_outer_vf) * h2o.rho,
               h2o.rho]
d2o_slds_dtail = [Si_mat.rho,
               SiO2.rho,
               surf_inner_vf * surf_head_inner.rho + (1-surf_inner_vf) * d2o.rho,
               1, # this will be updated by the vfp.
               1, # this will be updated by the vfp.
               surf_outer_vf * surf_head_outer.rho + (1-surf_outer_vf) * d2o.rho,
               d2o.rho]

# %%
# Define the dictionaries we use for the constraints

# D2O is defined first
d2o_constraint_ps = dict(
        surf_inner_head_vf=surf_inner_vf,
        surf_inner_head_sld=surf_head_inner.rho,
        surf_inner_tail_sld=surf_h_tail.rho,
        b_surf_inner_head=b_heads_inner,
        b_surf_inner_tail=b_htails, 
        surf_outer_head_vf=surf_outer_vf,
        surf_outer_head_sld=surf_head_outer.rho,
        surf_outer_tail_sld=surf_h_tail.rho,
        b_surf_outer_head=b_heads_outer,
        b_surf_outer_tail=b_htails,
        water_sld=d2o.rho
)

# Then we can make copies of this and overwrite the per-contrast parameters

h2o_constraint_ps = d2o_constraint_ps.copy()
h2o_constraint_ps['water_sld']=h2o.rho

d2o_dtail_constraint_ps = d2o_constraint_ps.copy()
d2o_dtail_constraint_ps['surf_inner_tail_sld']=surf_d_tail.rho
d2o_dtail_constraint_ps['surf_outer_tail_sld']=surf_d_tail.rho
d2o_dtail_constraint_ps['b_surf_outer_tail']=b_dtails
d2o_dtail_constraint_ps['b_surf_inner_tail']=b_dtails



# %%
# Assemble these into the VFP objects for each contrast:

d2o_vfp = vfp.refl1dVFP(nslds=d2o_slds,
                     thicknesses=thicknesses,
                     roughnesses=roughnesses,
                     sld_constraint=SLDConstraint(required_pars=d2o_constraint_ps))

h2o_vfp = vfp.refl1dVFP(nslds=h2o_slds,
                     thicknesses=thicknesses,
                     roughnesses=roughnesses,
                     sld_constraint=SLDConstraint(required_pars=h2o_constraint_ps))

d2o_dtails_vfp = vfp.refl1dVFP(nslds=d2o_slds_dtail,
                     thicknesses=thicknesses,
                     roughnesses=roughnesses,
                     sld_constraint=SLDConstraint(required_pars=d2o_dtail_constraint_ps))

# %% [markdown]
# ### We can see what the starting guess looks like

# %%
## Note: this might need updating with new plotting features.
d2o_vfp.plot()
plt.show()

# %% [markdown]
# ## Setup the Objective to Fit:

# %%
# Define our structures

d2o_vfp_struc = Si_mat | d2o_vfp | d2o
h2o_vfp_struc = Si_mat | h2o_vfp | h2o
d2o_dtails_vfp_struc = Si_mat | d2o_dtails_vfp | d2o

# %%
# Link the models and data into our Objectives

d2o_obj = Experiment(sample=d2o_vfp_struc, probe=d2o_probe, name='D2O')
h2o_obj = Experiment(sample=h2o_vfp_struc, probe=h2o_probe, name='H2O')
d2o_dt_obj = Experiment(sample=d2o_dtails_vfp_struc, probe=d2o_dtails_probe, name='D2O_DTAILS')
problem = FitProblem([d2o_obj, h2o_obj, d2o_dt_obj])

# %% [markdown]
# ## Check the setup for the fit optimisation:

# %%
# Check the parameters included in the optimisation and their bounds:
for p in problem.parameters:
    print(p.name, getattr(p, "bounds", getattr(p, "range", None)))


# %%
# check model looks reasonable with starting parameters

# %matplotlib inline
for mod in problem.models:
    plt.errorbar(x=mod.probe.Q, y=mod.probe.R, yerr=mod.probe.dR, linestyle='None', marker='.')
    plt.plot(*mod.reflectivity(), color='k', zorder=10)
plt.xscale('log')
plt.yscale('log')

# %% [markdown]
# ## Fit the objective.
#
# For clarity in the notebook we will run the fit optimisation in-line. For refl1d it is more common to save the model as a python file and run it either through the webview interface or via the command line. See the refl1d documentation for more information on this.

# %%
from bumps.fitters import fit

result = fit(problem, method="de", verbose=True, parallel=0, options=dict(steps=200))

# %% [markdown]
# ## Look at the Fit Outcome:

# %%
# Reflectivity plot from the fitted model
# %matplotlib inline
legend = [m.name for m in problem.models]
for idx, mod in enumerate(problem.models):
    plt.errorbar(x=mod.probe.Q, y=mod.probe.R, yerr=mod.probe.dR, linestyle='None', marker='.', label=legend[idx])
    plt.plot(*mod.reflectivity(), color='k', zorder=10)
plt.legend()
plt.title('Fitted model reflectivity')
plt.xlabel('Q (1/Å)')
plt.ylabel('R')
plt.xscale('log')
plt.yscale('log')

# %%
fig, ax = plt.subplots()
for vfp_obj, label in [(d2o_vfp, "D2O"), (h2o_vfp, "H2O"), (d2o_dtails_vfp, "D2O D-Tails")]:
    z, all_slds = vfp_obj.z_and_sld()
    ax.plot(z + vfp_obj.sld_offset(), all_slds[:, 0], label=label)  # sld_n
ax.set_ylabel("SLD / Å⁻² ×10⁻⁶")
ax.set_xlabel("Distance / Å")
ax.legend(frameon=False)
plt.title("SLD Profiles After Fit")
plt.show()


# %%
fig, ax = plt.subplots(1,3, figsize=(15,5))
fig.suptitle("Volume Fraction Profiles From Each Contrast After Fit")
for idx, (vfp_obj, lbl) in enumerate([(d2o_vfp, "D2O"), (h2o_vfp, "H2O"), (d2o_dtails_vfp, "D2O D-Tails")]):
    for material in range(len(vfp_obj.vfs_for_display()[0])):
        ax[idx].plot(vfp_obj.vfs_for_display()[0][material])
    ax[idx].set_title(f"{lbl} Contrast")
    ax[idx].set_xlabel("Distance / Å")
    ax[idx].set_ylabel("Volume Fraction")


# %% [markdown]
# This shows good agreement in the volume fractions for each component layer

# %% [markdown]
# ## Check the constraints on surface excess between contrasts has been achieved:

# %%
## Find vf from fit output
#print(d2o_vfp_struc.layer_parameters()['layers'][1]["nslds"][2].name)
#print(d2o_vfp_struc.layer_parameters()['layers'][1]["nslds"][6].name)
inner_vf = d2o_vfp_struc.layer_parameters()['layers'][1]["nslds"][2].value
outer_vf = d2o_vfp_struc.layer_parameters()['layers'][1]["nslds"][6].value
## SLDs are from original setup as don't vary in fit
inner_head_sld = surf_head_inner.rho.value
outer_head_sld = surf_head_outer.rho.value
htail_sld = surf_h_tail.rho.value
dtail_sld = surf_d_tail.rho.value

# %%
# Calculate the surface excesses from the fit output

from vfp.calc import integrate_vfp, init_demag

red_vfp, red_demag_vfp, idx, demag_arr = init_demag(
    d2o_vfp.tup_demag_locs,
    d2o_vfp.tup_demag_widths,
    d2o_vfp.tup_mslds,
    d2o_vfp.zeds,
    d2o_vfp._arrtotuple(d2o_vfp.vfp),
)

integrals = integrate_vfp(
    d2o_vfp.zeds,
    d2o_vfp.indices,
    d2o_vfp._arrtotuple(red_vfp),
    tuple([2, 3, 4, 5])
)


# calculate the vf from the parameters
surf_tail_vf_inner_h = (inner_vf * ((inner_head_sld * b_htails) / (b_heads_inner * htail_sld)) 
            * (integrals[0] / integrals[1])
)

surf_tail_vf_inner_d = (inner_vf * ((inner_head_sld * b_dtails) / (b_heads_inner * dtail_sld)) 
            * (integrals[0] / integrals[1])
)

surf_tail_vf_outer_h = (outer_vf * ((outer_head_sld * b_htails) / (b_heads_outer * htail_sld)) 
            * (integrals[3] / integrals[2])
)

surf_tail_vf_outer_d = (outer_vf * ((outer_head_sld * b_dtails) / (b_heads_outer * dtail_sld)) 
            * (integrals[3] / integrals[2])
)

print("-------------- Inner Comparison --------------")
print(f'Inner head surface excess = {(inner_vf * ((inner_head_sld * 1E-6) / (b_heads_inner*1E-5) * periodictable.constants.avogadro_number) * integrals[0]) / 1E20:.3e} mol m-2')
print(f'Inner tail surface excess from h = {(surf_tail_vf_inner_h * ((htail_sld * 1E-6) / (b_htails*1E-5) * periodictable.constants.avogadro_number) * integrals[1]) / 1E20:.3e} mol m-2')
print(f'Inner tail surface excess from d = {(surf_tail_vf_inner_d * ((dtail_sld * 1E-6) / (b_dtails*1E-5) * periodictable.constants.avogadro_number) * integrals[1]) / 1E20:.3e} mol m-2')
print("-------------- Outer Comparison --------------")
print(f'Outer head surface excess = {(outer_vf * ((outer_head_sld * 1E-6) / (b_heads_outer*1E-5) * periodictable.constants.avogadro_number) * integrals[3]) / 1E20:.3e} mol m-2')
print(f'Outer tail surface excess from h = {(surf_tail_vf_outer_h * ((htail_sld * 1E-6) / (b_htails*1E-5) * periodictable.constants.avogadro_number) * integrals[2]) / 1E20:.3e} mol m-2')
print(f'Outer tail surface excess from d = {(surf_tail_vf_outer_d * ((dtail_sld * 1E-6) / (b_dtails*1E-5) * periodictable.constants.avogadro_number) * integrals[2]) / 1E20:.3e} mol m-2')

# %% [markdown]
# ### These show the surface excess has been maintained across the 3 contrasts - Success!

# %%
