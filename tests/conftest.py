import numpy as np
import pytest
from refnx.analysis import Parameter
from refnx.reflect import SLD

rng = np.random.default_rng(seed=41)


@pytest.fixture
def sld_setup():
    si = SLD(2.07, name="si")
    sio2 = SLD(3.47, name="sio2")
    gmo = SLD(0.21, name="gmo")
    water = SLD(-0.56, name="water")
    hdod = SLD(-0.46, name="hdod")
    ddod = SLD(6.7, name="ddod")

    # lets vary the ddod but not the hdod.
    ddod.real.setp(vary=True, bounds=(6, 6.7))

    yield [si, sio2, gmo, water, hdod, ddod]


@pytest.fixture
def parameter_setup():
    sio2_thick = Parameter(20, name="sio2_thick", vary=True, bounds=(1, 30))
    inner_thick = Parameter(14, name="inner_thick", vary=True, bounds=(1, 30))
    outer_thick = Parameter(22, name="outer_thick", vary=True, bounds=(1, 30))
    si_sio2_rough = Parameter(
        3, name="si_sio2_rough", vary=True, bounds=(1, 5)
    )
    sio2_inner_rough = Parameter(
        4, name="sio2_inner_rough", vary=True, bounds=(1, 5)
    )
    inner_outer_rough = Parameter(
        4, name="inner_outer_rough", vary=True, bounds=(1, 5)
    )
    outer_solv_rough = Parameter(
        5, name="outer_solv_rough", vary=True, bounds=(1, 5)
    )
    inner_gmo_vf = Parameter(
        0.4, name="inner_gmo_vf", vary=True, bounds=(0, 1)
    )
    inner_solv_vf = Parameter(
        0.2, name="inner_solv_vf", vary=True, bounds=(0, 1)
    )
    outer_gmo_vf = Parameter(
        0.7, name="outer_gmo_vf", vary=True, bounds=(0, 1)
    )
    outer_solv_vf = Parameter(
        0.7, name="outer_solv_vf", vary=True, bounds=(0, 1)
    )
    ps = [
        sio2_thick,
        inner_thick,
        outer_thick,
        si_sio2_rough,
        sio2_inner_rough,
        inner_outer_rough,
        outer_solv_rough,
        inner_gmo_vf,
        inner_solv_vf,
        outer_gmo_vf,
        outer_solv_vf,
    ]
    yield ps


@pytest.fixture
def vfp_inputs(sld_setup: list[SLD], parameter_setup: list[Parameter]):
    si, sio2, gmo, water, hdod, ddod = sld_setup
    (
        sio2_thick,
        inner_thick,
        outer_thick,
        si_sio2_rough,
        sio2_inner_rough,
        inner_outer_rough,
        outer_solv_rough,
        inner_gmo_vf,
        inner_solv_vf,
        outer_gmo_vf,
        outer_solv_vf,
    ) = parameter_setup
    dd_gmo_nslds = [
        si.real,
        sio2.real,
        (1 - inner_solv_vf)
        * (gmo.real * inner_gmo_vf + (water.real * 1 - inner_gmo_vf))
        + ddod.real * inner_solv_vf,
        (1 - outer_solv_vf)
        * (gmo.real * outer_gmo_vf + (water.real * 1 - outer_gmo_vf))
        + ddod.real * outer_solv_vf,
        ddod.real,
    ]

    list_of_thicknesses_gmo = [0, sio2_thick, inner_thick, outer_thick]
    list_of_roughnesses_gmo = [
        si_sio2_rough,
        sio2_inner_rough,
        inner_outer_rough,
        outer_solv_rough,
    ]

    yield dd_gmo_nslds, list_of_thicknesses_gmo, list_of_roughnesses_gmo


@pytest.fixture
def materials_by_layer_setup(parameter_setup):
    (
        sio2_thick,
        inner_thick,
        outer_thick,
        si_sio2_rough,
        sio2_inner_rough,
        inner_outer_rough,
        outer_solv_rough,
        inner_gmo_vf,
        inner_solv_vf,
        outer_gmo_vf,
        outer_solv_vf,
    ) = parameter_setup
    materials_by_layer = {
        0: {"Si": 1},
        1: {r"$\mathrm{SiO}_2$": 1},
        2: {
            "water": (1 - inner_solv_vf) * (1 - inner_gmo_vf),
            "gmo": (1 - inner_solv_vf) * inner_gmo_vf,
            "ddod": inner_solv_vf,
        },
        3: {
            "water": (1 - outer_solv_vf) * (1 - outer_gmo_vf),
            "gmo": (1 - outer_solv_vf) * outer_gmo_vf,
            "ddod": outer_solv_vf,
        },
        4: {"ddod": 1},
    }
    yield materials_by_layer


@pytest.fixture
def posterior_samples_setup():
    posterior_samples = {
        "si_sio2_rough": rng.normal(loc=3, scale=0.25, size=300),
        "sio2_thick": rng.normal(loc=20, scale=0.4, size=300),
        "inner_thick": rng.normal(loc=14, scale=1, size=300),
        "outer_thick": rng.normal(loc=22, scale=2, size=300),
        "inner_gmo_vf": rng.normal(loc=0.4, scale=0.02, size=300),
        "inner_solv_vf": rng.normal(loc=0.2, scale=0.023, size=300),
        "outer_gmo_vf": rng.normal(loc=0.7, scale=0.03, size=300),
        "outer_solv_vf": rng.normal(loc=0.7, scale=0.038, size=300),
    }
    yield posterior_samples
