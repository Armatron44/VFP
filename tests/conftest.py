import numpy as np
import pytest
from refnx.analysis import Parameter
from refnx.reflect import SLD

rng = np.random.default_rng(seed=41)
surface_rng = np.random.default_rng(seed=42)

# fixutres for plotting


@pytest.fixture
def sld_setup():
    fronting_sld = SLD(2, name="fronting_sld")
    lay1nsld = SLD(3.5, name="lay1nsld")
    interf_mat1sld = SLD(0.2, name="interf_mat1sld")
    interf_mat2sld = SLD(-0.5, name="interf_mat2sld")
    backing_sld = SLD(6.7, name="backing_sld")

    backing_sld.real.setp(vary=True, bounds=(6, 6.7))

    lay1msld = SLD(2.3, name="lay1msld")

    nslds = [
        fronting_sld,
        lay1nsld,
        interf_mat1sld,
        interf_mat2sld,
        backing_sld,
    ]

    mslds = [lay1msld]

    islds = [0.0, 0.2, 0.4, 0.3, 0]

    yield [nslds, mslds, islds]


@pytest.fixture
def parameter_setup():
    lay1_thick = Parameter(20, name="lay1_thick", vary=True, bounds=(1, 30))
    lay2_thick = Parameter(14, name="lay2_thick", vary=True, bounds=(1, 30))
    lay3_thick = Parameter(22, name="lay3_thick", vary=True, bounds=(1, 30))
    fronting_lay1_rough = Parameter(
        3, name="fronting_lay1_rough", vary=True, bounds=(1, 5)
    )
    lay1_lay2_rough = Parameter(
        4, name="lay1_lay2_rough", vary=True, bounds=(1, 5)
    )
    lay2_lay3_rough = Parameter(
        4, name="lay2_lay3_rough", vary=True, bounds=(1, 5)
    )
    lay3_backing_rough = Parameter(
        5, name="lay3_backing_rough", vary=True, bounds=(1, 5)
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
    ps = [
        lay1_thick,
        lay2_thick,
        lay3_thick,
        fronting_lay1_rough,
        lay1_lay2_rough,
        lay2_lay3_rough,
        lay3_backing_rough,
        layer2_interf_mat1_vf,
        layer2_backing_vf,
        layer3_interf_mat1_vf,
        layer3_backing_vf,
    ]
    yield ps


@pytest.fixture
def vfp_inputs(
    sld_setup: list[list[SLD], list[float]], parameter_setup: list[Parameter]
):
    nslds, mslds, islds = sld_setup
    (fronting_sld, lay1nsld, interf_mat1sld, interf_mat2sld, backing_sld) = (
        nslds
    )
    (lay1msld,) = mslds
    (
        lay1_thick,
        lay2_thick,
        lay3_thick,
        fronting_lay1_rough,
        lay1_lay2_rough,
        lay2_lay3_rough,
        lay3_backing_rough,
        layer2_interf_mat1_vf,
        layer2_backing_vf,
        layer3_interf_mat1_vf,
        layer3_backing_vf,
    ) = parameter_setup
    vfp_nslds = [
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

    vfp_mslds = [0, lay1msld.real, 0, 0, 0]

    list_of_thicknesses_gmo = [0, lay1_thick, lay2_thick, lay3_thick]
    list_of_roughnesses_gmo = [
        fronting_lay1_rough,
        lay1_lay2_rough,
        lay2_lay3_rough,
        lay3_backing_rough,
    ]

    yield (
        vfp_nslds,
        vfp_mslds,
        islds,
        list_of_thicknesses_gmo,
        list_of_roughnesses_gmo,
    )


@pytest.fixture
def materials_by_layer_setup(parameter_setup):
    (
        lay1_thick,
        lay2_thick,
        lay3_thick,
        fronting_lay1_rough,
        lay1_lay2_rough,
        lay2_lay3_rough,
        lay3_backing_rough,
        layer2_interf_mat1_vf,
        layer2_backing_vf,
        layer3_interf_mat1_vf,
        layer3_backing_vf,
    ) = parameter_setup
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
    yield materials_by_layer


@pytest.fixture(scope="session")
def posterior_samples_setup():
    posterior_samples = {
        "fronting_lay1_rough": rng.normal(loc=3, scale=0.25, size=300),
        "lay1_lay2_rough": rng.normal(loc=4, scale=0.3, size=300),
        "lay2_lay3_rough": rng.normal(loc=4, scale=0.4, size=300),
        "lay3_backing_rough": rng.normal(loc=5, scale=0.5, size=300),
        "lay1_thick": rng.normal(loc=20, scale=0.4, size=300),
        "lay2_thick": rng.normal(loc=14, scale=1, size=300),
        "lay3_thick": rng.normal(loc=22, scale=2, size=300),
        "layer2_interf_mat1_vf": rng.normal(loc=0.4, scale=0.02, size=300),
        "layer2_backing_vf": rng.normal(loc=0.2, scale=0.023, size=300),
        "layer3_interf_mat1_vf": rng.normal(loc=0.7, scale=0.03, size=300),
        "layer3_backing_vf": rng.normal(loc=0.7, scale=0.038, size=300),
    }
    yield posterior_samples


@pytest.fixture
def plot_kwargs(posterior_samples_setup, materials_by_layer_setup):
    posterior_samples = posterior_samples_setup
    materials_by_layer = materials_by_layer_setup

    base_plot_kwargs = dict(posterior_samples=posterior_samples)
    addn_plot_kwargs = [
        {
            "vfp_plot_kwargs": {"layer_materials": materials_by_layer},
            "sld_plot_kwargs": {"microslice": True, "total_sld": True},
            "surface_plot_kwargs": {"surface_rng": surface_rng},
        },
        {
            "align_at": 2,
            "vfp_plot_kwargs": {"layer_materials": materials_by_layer},
            "sld_plot_kwargs": {"microslice": True, "total_sld": False},
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
        {
            "align_at": 3,
            "plots_required": ["surfaces", "vfp"],
            "surface_plot_kwargs": {"surface_rng": surface_rng},
        },
    ]

    keys = [
        "allplots_layermats_mslice_tsld",
        "allplots_layermats_mslice_align_at_2",
        "sldvfpplots_layermats",
        "vfp_post",
        "surfacesvfp",
        "surfacesvfp_align_at_3",
    ]

    final_plot_kwargs = {
        key: base_plot_kwargs | akwargs
        for key, akwargs in zip(keys, addn_plot_kwargs, strict=False)
    }
    yield final_plot_kwargs


# fixtures for test_calc
@pytest.fixture()
def init_standard_sample():
    # set up some quick standard test parameters.
    # 4 layers + use some non-integer values.
    lot = [0, 19.7, 50, 30]
    lor = [3.1, 5, 7.3, 6]
    nslds = [0, 6, 4, 3.47, 2.07]
    mslds = [0, 0, 3, 0, 0]
    locs = [1, 25]
    widths = [1, 5]
    conformal = [0, 0, 0, 0]
    dict_res = {
        "thicks": lot,
        "roughs": lor,
        "nslds": nslds,
        "mslds": mslds,
        "locs": locs,
        "widths": widths,
        "conformal": conformal,
    }
    yield dict_res


@pytest.fixture()
def init_standard_sample_two():
    # set up some quick standard test parameters.
    # 4 layers + use some non-integer values.
    lot = [0, 32.8, 16.3, 24.6]
    lor = [4.1, 6, 7, 3]
    nslds = [5.9, 1.2, 3.6, 0.1, -0.46]
    mslds = [0, 1.1, 0, 2.2, 0]
    locs = [1, 23, 18, 22]
    widths = [1.3, 7, 4.2, 5.1]
    conformal = [0, 1, 0, 0]
    dict_res = {
        "thicks": lot,
        "roughs": lor,
        "nslds": nslds,
        "mslds": mslds,
        "locs": locs,
        "widths": widths,
        "conformal": conformal,
    }
    yield dict_res
