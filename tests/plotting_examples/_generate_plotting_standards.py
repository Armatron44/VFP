"""Script to generate plots to test against.
These plots should not be regenerated unless some plotting change is required.
"""

import pathlib

import numpy as np
from matplotlib.figure import Figure
from refnx.analysis import Parameter
from refnx.reflect import SLD

from vfp import vfp


def figs_to_arr(figs: list[Figure]) -> list[np.ndarray]:
    image_arrs = []
    for fig in figs:
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        rgba_buffer = fig.canvas.buffer_rgba()
        image_array = np.asarray(rgba_buffer, dtype=np.uint8)
        image_array = image_array.reshape(int(height), int(width), 4)
        image_arrs.append(image_array)
    return image_arrs


def save_all_plot_data(image_arrs: dict[str, np.ndarray]):
    np.savez_compressed(
        file=pathlib.Path(__file__).parent
        / pathlib.Path(r"plotting_examples.npz"),
        **image_arrs,
    )


fronting_sld = SLD(2, name="fronting_sld")
lay1nsld = SLD(3.5, name="lay1nsld")
interf_mat1sld = SLD(0.2, name="interf_mat1sld")
interf_mat2sld = SLD(-0.5, name="interf_mat2sld")
backing_sld = SLD(6.7, name="backing_sld")

lay1msld = SLD(2.3, name="lay1msld")

backing_sld.real.setp(vary=True, bounds=(6, 6.7))

# define thickness and roughness parameters.
lay1_thick = Parameter(20, name="lay1_thick", vary=True, bounds=(1, 30))
lay2_thick = Parameter(14, name="lay2_thick", vary=True, bounds=(1, 30))
lay3_thick = Parameter(22, name="lay3_thick", vary=True, bounds=(1, 30))

fronting_lay1_rough = Parameter(
    3, name="fronting_lay1_rough", vary=True, bounds=(1, 10)
)
lay1_lay2_rough = Parameter(
    4, name="lay1_lay2_rough", vary=True, bounds=(1, 10)
)
lay2_lay3_rough = Parameter(
    4, name="lay2_lay3_rough", vary=True, bounds=(1, 10)
)
lay3_backing_rough = Parameter(
    5, name="lay3_backing_rough", vary=True, bounds=(1, 10)
)

layer2_interf_mat1_vf = Parameter(
    0.4, name="layer2_interf_mat1_vf", vary=True, bounds=(0, 1)
)
layer2_backing_vf = Parameter(
    0.2, name="layer2_backing_vf", vary=True, bounds=(0, 1)
)

layer3_interf_mat1_vf = Parameter(
    0.7, name="layer3_interf_mat1_vf", vary=True, bounds=(0, 1)
)
layer3_backing_vf = Parameter(
    0.7, name="layer3_backing_vf", vary=True, bounds=(0, 1)
)

materials_by_layer = {
    0: {"fronting": 1},
    1: {"lay1": 1},
    2: {
        "mat2": (1 - layer2_backing_vf) * (1 - layer2_interf_mat1_vf),
        "mat1": (1 - layer2_backing_vf) * layer2_interf_mat1_vf,
        "backing": layer2_backing_vf,
    },
    3: {
        "mat2": (1 - layer3_backing_vf) * (1 - layer3_interf_mat1_vf),
        "mat1": (1 - layer3_backing_vf) * layer3_interf_mat1_vf,
        "backing": layer3_backing_vf,
    },
    4: {"backing": 1},
}

posterior_rng = np.random.default_rng(seed=41)

posterior_samples = {
    "fronting_lay1_rough": posterior_rng.normal(loc=3, scale=0.25, size=300),
    "lay1_lay2_rough": posterior_rng.normal(loc=4, scale=0.3, size=300),
    "lay2_lay3_rough": posterior_rng.normal(loc=4, scale=0.4, size=300),
    "lay3_backing_rough": posterior_rng.normal(loc=5, scale=0.5, size=300),
    "lay1_thick": posterior_rng.normal(loc=20, scale=0.4, size=300),
    "lay2_thick": posterior_rng.normal(loc=14, scale=1, size=300),
    "lay3_thick": posterior_rng.normal(loc=22, scale=2, size=300),
    "layer2_interf_mat1_vf": posterior_rng.normal(
        loc=0.4, scale=0.02, size=300
    ),
    "layer2_backing_vf": posterior_rng.normal(loc=0.2, scale=0.023, size=300),
    "layer3_interf_mat1_vf": posterior_rng.normal(
        loc=0.7, scale=0.03, size=300
    ),
    "layer3_backing_vf": posterior_rng.normal(loc=0.7, scale=0.038, size=300),
}

nslds = [
    fronting_sld.real,
    lay1nsld.real,
    (1 - layer2_backing_vf)
    * (
        interf_mat1sld.real * layer2_interf_mat1_vf
        + (interf_mat2sld.real * 1 - layer2_interf_mat1_vf)
    )
    + backing_sld.real * layer2_backing_vf,
    (1 - layer3_backing_vf)
    * (
        interf_mat1sld.real * layer3_interf_mat1_vf
        + (interf_mat2sld.real * 1 - layer3_interf_mat1_vf)
    )
    + backing_sld.real * layer3_backing_vf,
    backing_sld.real,
]

mslds = [0, lay1msld.real, 0, 0, 0]

islds = [0.0, 0.2, 0.4, 0.3, 0]

list_of_thicknesses_gmo = [0, lay1_thick, lay2_thick, lay3_thick]
# define the width the of the interfaces between the materials.
list_of_roughnesses_gmo = [
    fronting_lay1_rough,
    lay1_lay2_rough,
    lay2_lay3_rough,
    lay3_backing_rough,
]

base_kwargs = dict(
    nslds=nslds,
    thicknesses=list_of_thicknesses_gmo,
    roughnesses=list_of_roughnesses_gmo,
    mslds=mslds,
    islds=islds,
)

addn_kwargs = [
    {"spin_state": "up"},
    {"orientation": "back", "spin_state": "up"},
    {"max_delta_z": 0.1, "spin_state": "down"},
]

final_kwargs = [base_kwargs | akwargs for akwargs in addn_kwargs]
vfps = [vfp.refnxVFP(**f_kwarg) for f_kwarg in final_kwargs]

surface_rng = np.random.default_rng(seed=42)

base_plot_kwargs = dict(posterior_samples=posterior_samples)
addn_plot_kwargs = [
    {
        "vfp_plot_kwargs": {"layer_materials": materials_by_layer},
        "sld_plot_kwargs": {"microslice": True, "total_sld": True},
        "surface_plot_kwargs": {"surface_rng": surface_rng},
    },
    {
        "plots_required": ["sld", "vfp"],
        "vfp_plot_kwargs": {"layer_materials": materials_by_layer},
        "sld_plot_kwargs": {"microslice": False, "total_sld": False},
    },
    {"plots_required": ["vfp"], "posterior_samples": None},
    {
        "plots_required": ["surfaces", "vfp"],
        "surface_plot_kwargs": {"surface_rng": surface_rng},
    },
]
final_plot_kwargs = [
    base_plot_kwargs | akwargs for akwargs in addn_plot_kwargs
]


def plot_vfp(
    vfps: list[vfp.refnxVFP], final_plot_kwargs: list[dict]
) -> list[Figure]:
    figs = []
    for v in vfps:
        for f_p_kwarg in final_plot_kwargs:
            fig, _ = v.plot(**f_p_kwarg)
            figs.append(fig)
    return figs


figs = plot_vfp(vfps, final_plot_kwargs)

image_arrs = figs_to_arr(figs)
descrips = [
    "front_allplots_layermats_mslice_tsld",
    "front_sldvfpplots_layermats",
    "front_vfp_post",
    "front_surfacesvfp",
    "back_ssup_allplots_layermats_mslice_tsld",
    "back_ssup_sldvfpplots_layermats",
    "back_ssup_vfp_post",
    "back_ssup_surfacesvfp",
    "front_mdz_01_ss_down_allplots_layermats_mslice_tsld",
    "front_mdz_01_ss_down_sldvfpplots_layermats",
    "front_mdz_01_ss_down_vfp_post",
    "front_mdz_01_ss_down_surfacesvfp",
]

# plt.show()
plots_dict = {
    descp: im_arr for descp, im_arr in zip(descrips, image_arrs, strict=False)
}
save_all_plot_data(plots_dict)
