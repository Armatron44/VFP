"""
Tests VFP and BaseVFP methods.
"""

from functools import partial

import numpy as np
import pytest
from scipy.stats import norm

from vfp import VFP, refl1dVFP, refnxVFP
from vfp.vfp_typing import ParameterLike

EPS = np.finfo(float).eps

"""
Test all permutations of conformal interfaces with a 3 layer interface.
TODO: neaten up!
"""

# standard VFP layers, no conformality
# calculate vf profiles for each layer.
start = -17  # -5 - (4 * 3) = -17
end = 86  # 5 + (0 + 20 + 30 + 15) + 4 * 4 = 86
z = np.linspace(start, end, int((end - start) / 0.5) + 1)
fronting = 1 - norm.cdf(
    z, loc=0, scale=3
)  # fronting is constant, cannot be conformal.
first_lay_e0 = norm.cdf(z, loc=0, scale=3) * (
    1 - norm.cdf(z, loc=20, scale=2)
)
second_lay_e0 = (
    norm.cdf(z, loc=0, scale=3)
    * norm.cdf(z, loc=20, scale=2)
    * (1 - norm.cdf(z, loc=20 + 30, scale=4))
)
third_lay_e0 = (
    norm.cdf(z, loc=0, scale=3)
    * norm.cdf(z, loc=20, scale=2)
    * (norm.cdf(z, loc=20 + 30, scale=4))
    * (1 - (norm.cdf(z, loc=20 + 30 + 15, scale=1.5)))
)
backing_e0 = 1 - np.sum(
    (fronting, first_lay_e0, second_lay_e0, third_lay_e0), axis=0
)

# conformal examples
# conformal = (0, 1, 0, 0)
# conformal interfaces are = interface before but translated in z.
f0_st1 = norm.cdf(
    z, loc=0 + 20, scale=3
)  # name means interface 0 (f0), (s)hifted by (t)hickness of layer (1)
# volume fraction of layer with conformal interface is
# (1-conformal interface) - everything before this layer
first_lay_e1 = (1 - f0_st1) - fronting
# vf of layers after conformal interface is as we calc above but now the
# conformal interface is where we calc from.
second_lay_e1 = f0_st1 * (1 - norm.cdf(z, loc=20 + 30, scale=4))
third_lay_e1 = (
    f0_st1
    * norm.cdf(z, loc=20 + 30, scale=4)
    * (1 - (norm.cdf(z, loc=20 + 30 + 15, scale=1.5)))
)
backing_e1 = 1 - np.sum(
    (fronting, first_lay_e1, second_lay_e1, third_lay_e1), axis=0
)

# conformal = (0, 0, 1, 0)
# our largest roughness has changed so we need to recalc z.
end = 82  # 5 + (0 + 20 + 30 + 15) + 4 * 3 = 82
z = np.linspace(start, end, int((end - start) / 0.5) + 1)
# calc non-conformal
fronting_e2 = 1 - norm.cdf(z, loc=0, scale=3)
first_lay_e2 = norm.cdf(z, loc=0, scale=3) * (
    1 - norm.cdf(z, loc=20, scale=2)
)
# second_lay's interface is conformal to the combination of all interfaces
# before it, so translate the first and second layer interfaces by the
# thickness of layer 2.
f0_st2 = norm.cdf(z, loc=0 + 30, scale=3)
f1_st2 = norm.cdf(z, loc=0 + 20 + 30, scale=2)
second_lay_e2 = (1 - (f0_st2 * f1_st2)) - (fronting_e2 + first_lay_e2)
third_lay_e2 = (f0_st2 * f1_st2) * (
    1 - (norm.cdf(z, loc=20 + 30 + 15, scale=1.5))
)
backing_e2 = 1 - np.sum(
    (fronting_e2, first_lay_e2, second_lay_e2, third_lay_e2), axis=0
)

# conformal = (0, 0, 0, 1)
# our largest roughness has changed so we need to recalc z.
end = 86  # 5 + (0 + 20 + 30 + 15) + 4 * 4 = 86
z = np.linspace(start, end, int((end - start) / 0.5) + 1)
# calc non-conformal
fronting_e3 = 1 - norm.cdf(z, loc=0, scale=3)
first_lay_e3 = norm.cdf(z, loc=0, scale=3) * (
    1 - norm.cdf(z, loc=20, scale=2)
)
second_lay_e3 = (
    norm.cdf(z, loc=0, scale=3)
    * norm.cdf(z, loc=20, scale=2)
    * (1 - norm.cdf(z, loc=20 + 30, scale=4))
)
f0_st3 = norm.cdf(z, loc=0 + 15, scale=3)
f1_st3 = norm.cdf(z, loc=0 + 20 + 15, scale=2)
f2_st3 = norm.cdf(z, loc=0 + 20 + 30 + 15, scale=4)
third_lay_e3 = (1 - (f0_st3 * f1_st3 * f2_st3)) - (
    fronting_e3 + first_lay_e3 + second_lay_e3
)
backing_e3 = 1 - np.sum(
    (fronting_e3, first_lay_e3, second_lay_e3, third_lay_e3), axis=0
)

# conformal = (0, 1, 1, 0)
end = 82  # 5 + (0 + 20 + 30 + 15) + 4 * 3 = 82
z = np.linspace(start, end, int((end - start) / 0.5) + 1)
fronting_e4 = 1 - norm.cdf(z, loc=0, scale=3)
f0_st1 = norm.cdf(z, loc=0 + 20, scale=3)
f0_st1t2 = norm.cdf(z, loc=0 + 20 + 30, scale=3)

first_lay_e4 = (1 - f0_st1) - fronting_e4
second_lay_e4 = (1 - f0_st1t2) - (first_lay_e4 + fronting_e4)
third_lay_e4 = f0_st1t2 * (1 - norm.cdf(z, loc=20 + 30 + 15, scale=1.5))
backing_e4 = 1 - np.sum(
    (fronting_e4, first_lay_e4, second_lay_e4, third_lay_e4), axis=0
)

# conformal = (0, 0, 1, 1)
end = 82  # 5 + (0 + 20 + 30 + 15) + 4 * 3 = 82
z = np.linspace(start, end, int((end - start) / 0.5) + 1)
fronting_e5 = 1 - norm.cdf(z, loc=0, scale=3)
first_lay_e5 = norm.cdf(z, loc=0, scale=3) * (
    1 - norm.cdf(z, loc=20, scale=2)
)

f0_st2 = norm.cdf(z, loc=0 + 30, scale=3)
f1_st2 = norm.cdf(z, loc=0 + 20 + 30, scale=2)
f0_st2t3 = norm.cdf(z, loc=0 + 30 + 15, scale=3)
f1_st2t3 = norm.cdf(z, loc=0 + 20 + 30 + 15, scale=2)

second_lay_e5 = (1 - (f0_st2 * f1_st2)) - (fronting_e5 + first_lay_e5)
third_lay_e5 = (1 - (f0_st2t3 * f1_st2t3)) - (
    fronting_e5 + first_lay_e5 + second_lay_e5
)
backing_e5 = 1 - np.sum(
    (fronting_e5, first_lay_e5, second_lay_e5, third_lay_e5), axis=0
)

# conformal = (0, 1, 0, 1)
end = 86  # 5 + (0 + 20 + 30 + 15) + 4 * 4 = 86
z = np.linspace(start, end, int((end - start) / 0.5) + 1)
fronting_e6 = 1 - norm.cdf(z, loc=0, scale=3)
f0_st1 = norm.cdf(z, loc=0 + 20, scale=3)
f2_st3 = norm.cdf(z, loc=20 + 30 + 15, scale=4)

first_lay_e6 = (1 - f0_st1) - fronting_e6

second_lay_e6 = f0_st1 * (1 - norm.cdf(z, loc=20 + 30, scale=4))
third_lay_e6 = (1 - f2_st3) - (fronting_e6 + first_lay_e6 + second_lay_e6)
backing_e6 = 1 - np.sum(
    (fronting_e6, first_lay_e6, second_lay_e6, third_lay_e6), axis=0
)

# conformal = (0, 1, 1, 1)
end = 82  # 5 + (0 + 20 + 30 + 15) + 4 * 3 = 82
z = np.linspace(start, end, int((end - start) / 0.5) + 1)
fronting_e7 = 1 - norm.cdf(z, loc=0, scale=3)
f0_st1 = norm.cdf(z, loc=0 + 20, scale=3)
f0_st1t2 = norm.cdf(z, loc=0 + 20 + 30, scale=3)
f0_st1t2t3 = norm.cdf(z, loc=0 + 20 + 30 + 15, scale=3)

first_lay_e7 = (1 - f0_st1) - fronting_e7
second_lay_e7 = (1 - f0_st1t2) - (fronting_e7 + first_lay_e7)
third_lay_e7 = (1 - f0_st1t2t3) - (fronting_e7 + first_lay_e7 + second_lay_e7)
backing_e7 = 1 - np.sum(
    (fronting_e7, first_lay_e7, second_lay_e7, third_lay_e7), axis=0
)

standard_examples_conformal = [
    (
        (3, 2, 4, 1.5),
        np.vstack(
            (fronting, first_lay_e0, second_lay_e0, third_lay_e0, backing_e0)
        ),
    ),
    (
        (3, "conformal", 4, 1.5),
        np.vstack(
            (fronting, first_lay_e1, second_lay_e1, third_lay_e1, backing_e1)
        ),
    ),
    (
        (3, 2, "conformal", 1.5),
        np.vstack(
            (
                fronting_e2,
                first_lay_e2,
                second_lay_e2,
                third_lay_e2,
                backing_e2,
            )
        ),
    ),
    (
        (3, 2, 4, "conformal"),
        np.vstack(
            (
                fronting_e3,
                first_lay_e3,
                second_lay_e3,
                third_lay_e3,
                backing_e3,
            )
        ),
    ),
    (
        (3, "conformal", "conformal", 1.5),
        np.vstack(
            (
                fronting_e4,
                first_lay_e4,
                second_lay_e4,
                third_lay_e4,
                backing_e4,
            )
        ),
    ),
    (
        (3, 2, "conformal", "conformal"),
        np.vstack(
            (
                fronting_e5,
                first_lay_e5,
                second_lay_e5,
                third_lay_e5,
                backing_e5,
            )
        ),
    ),
    (
        (3, "conformal", 4, "conformal"),
        np.vstack(
            (
                fronting_e6,
                first_lay_e6,
                second_lay_e6,
                third_lay_e6,
                backing_e6,
            )
        ),
    ),
    (
        (3, "conformal", "conformal", "conformal"),
        np.vstack(
            (
                fronting_e7,
                first_lay_e7,
                second_lay_e7,
                third_lay_e7,
                backing_e7,
            )
        ),
    ),
]


@pytest.mark.parametrize(
    "roughnesses, expected_result", standard_examples_conformal
)
def test_vfps_conformal(
    roughnesses: list[ParameterLike | str], expected_result: np.ndarray
):
    """
    Test vfps with different conformality.
    Tests all three vfps give same answer & if result is same to known result.
    """
    thicknesses = (0, 20, 30, 15)
    slds = (2, 3, 0, 1.5, 6.7)
    vfp = VFP(nslds=slds, thicknesses=thicknesses, roughnesses=roughnesses)
    refnx_vfp = refnxVFP(
        nslds=slds, thicknesses=thicknesses, roughnesses=roughnesses
    )
    refl1d_vfp = refl1dVFP(
        nslds=slds, thicknesses=thicknesses, roughnesses=roughnesses
    )

    # check vfp types are same
    np.testing.assert_allclose(vfp.vfp, refnx_vfp.vfp)
    np.testing.assert_allclose(vfp.vfp, refl1d_vfp.vfp)
    np.testing.assert_allclose(refnx_vfp.vfp, refl1d_vfp.vfp)

    # check vfp attr is the same expected result
    for v in [vfp, refnx_vfp, refl1d_vfp]:
        np.testing.assert_allclose(v.vfp, expected_result, atol=EPS)


# test orientation option gives correct SLD profile for simple model
# and a model with conformal interface.
nslds = (0.1, 3, 4.5)
thicknesses = (0, 20)
roughnesses = (3, 2)
msld = (0, 1, 2.3)
isld = (7.5e-3, 1e-4, 3e-2)

start = -17  # -5 - (4 * 3) = -17
end = 37  # 5 + (0 + 20) + 4 * 4.5 = 37
z = np.linspace(start, end, int((end - start) / 0.5) + 1)
fronting_sld = (1 - norm.cdf(z, loc=0, scale=3)) * 0.1
fronting_msld = (1 - norm.cdf(z, loc=0, scale=3)) * 0
fronting_isld = (1 - norm.cdf(z, loc=0, scale=3)) * 7.5 * 1e-3
first_lay_sld = 3 * (
    norm.cdf(z, loc=0, scale=3) * (1 - norm.cdf(z, loc=20, scale=2))
)
first_lay_msld = 1 * (
    norm.cdf(z, loc=0, scale=3) * (1 - norm.cdf(z, loc=20, scale=2))
)
first_lay_isld = 1e-4 * (
    norm.cdf(z, loc=0, scale=3) * (1 - norm.cdf(z, loc=20, scale=2))
)
backing_sld = 4.5 * (
    norm.cdf(z, loc=0, scale=3) * norm.cdf(z, loc=20, scale=2)
)
backing_msld = 2.3 * (
    norm.cdf(z, loc=0, scale=3) * norm.cdf(z, loc=20, scale=2)
)
backing_isld = 3e-2 * (
    norm.cdf(z, loc=0, scale=3) * norm.cdf(z, loc=20, scale=2)
)

orientation_expected_result_front_nospin = np.vstack(
    (
        z,
        np.sum((fronting_sld, first_lay_sld, backing_sld), axis=0),
        np.sum((fronting_isld, first_lay_isld, backing_isld), axis=0),
        np.sum(
            (
                np.zeros_like(fronting_msld),
                np.zeros_like(first_lay_msld),
                np.zeros_like(backing_msld),
            ),
            axis=0,
        ),
    )
)

orientation_expected_result_back_nospin = np.vstack(
    (
        -z,  # orientation = back - flip z
        np.sum((fronting_sld, first_lay_sld, backing_sld), axis=0),
        np.sum((fronting_isld, first_lay_isld, backing_isld), axis=0),
        np.sum(
            (
                np.zeros_like(fronting_msld),
                np.zeros_like(first_lay_msld),
                np.zeros_like(backing_msld),
            ),
            axis=0,
        ),
    )
)

orientation_expected_result_front = np.vstack(
    (
        z,
        np.sum((fronting_sld, first_lay_sld, backing_sld), axis=0),
        np.sum((fronting_isld, first_lay_isld, backing_isld), axis=0),
        np.sum((fronting_msld, first_lay_msld, backing_msld), axis=0),
    )
)

orientation_expected_result_back = np.vstack(
    (
        -z,  # orientation = back - flip z.
        np.sum((fronting_sld, first_lay_sld, backing_sld), axis=0),
        np.sum((fronting_isld, first_lay_isld, backing_isld), axis=0),
        np.sum((fronting_msld, first_lay_msld, backing_msld), axis=0),
    )
)

standard_examples_orientation = [
    (
        nslds,
        thicknesses,
        roughnesses,
        "front",
        (0, 0, 0),
        isld,
        "none",
        orientation_expected_result_front_nospin,
    ),
    (
        nslds,
        thicknesses,
        roughnesses,
        "back",
        (0, 0, 0),
        isld,
        "none",
        orientation_expected_result_back_nospin,
    ),
    (
        nslds,
        thicknesses,
        roughnesses,
        "front",
        msld,
        isld,
        "up",
        orientation_expected_result_front,
    ),
    (
        nslds,
        thicknesses,
        roughnesses,
        "back",
        msld,
        isld,
        "up",
        orientation_expected_result_back,
    ),
    (
        nslds,
        thicknesses,
        roughnesses,
        "front",
        msld,
        isld,
        "down",
        orientation_expected_result_front,
    ),
    (
        nslds,
        thicknesses,
        roughnesses,
        "back",
        msld,
        isld,
        "down",
        orientation_expected_result_back,
    ),
]


@pytest.mark.parametrize(
    "nslds, thicknesses, roughnesses, orientation, mslds, islds, spin_state"
    ", expected_result",
    standard_examples_orientation,
)
def test_slds_orientation(  # noqa : PLR0913
    nslds: tuple,
    thicknesses: tuple,
    roughnesses: tuple,
    orientation: str,
    mslds: tuple,
    islds: tuple,
    spin_state: str,
    expected_result: float,
):
    vfp_kwargs = locals()
    del vfp_kwargs["expected_result"]

    vfp = VFP(**vfp_kwargs)
    refnx_vfp = refnxVFP(**vfp_kwargs)
    refl1d_vfp = refl1dVFP(**vfp_kwargs)

    np.testing.assert_allclose(
        vfp.z_and_sld(reduced=False)[1], refnx_vfp.z_and_sld(reduced=False)[1]
    )
    np.testing.assert_allclose(
        vfp.z_and_sld(reduced=False)[1],
        refl1d_vfp.z_and_sld(reduced=False)[1],
    )
    np.testing.assert_allclose(
        refnx_vfp.z_and_sld(reduced=False)[1],
        refl1d_vfp.z_and_sld(reduced=False)[1],
    )

    # check vfp attr is the same expected result
    for v in [vfp, refnx_vfp, refl1d_vfp]:
        np.testing.assert_allclose(
            v.z_and_sld(reduced=False)[0], expected_result[0], atol=EPS
        )
        np.testing.assert_allclose(
            v.z_and_sld(reduced=False)[1], expected_result[1:].T, atol=EPS
        )


mask = np.ones_like(z, dtype=bool)
# 8 * 0.5 + 0.5 = 4.5
mask[1:9] = False  # known points that are removed when reduced.
mask[92:108] = False  # 16 * 0.5 + 0.5

z_and_slds_reduced_expected_result = np.vstack(
    (
        z[mask],
        np.sum(
            (fronting_sld[mask], first_lay_sld[mask], backing_sld[mask]),
            axis=0,
        ),
        np.sum(
            (fronting_isld[mask], first_lay_isld[mask], backing_isld[mask]),
            axis=0,
        ),
        np.sum(
            (fronting_msld[mask], first_lay_msld[mask], backing_msld[mask]),
            axis=0,
        ),
    )
)

z_and_slds_reduced_nospin_expected_result = np.vstack(
    (
        z[mask],
        np.sum(
            (fronting_sld[mask], first_lay_sld[mask], backing_sld[mask]),
            axis=0,
        ),
        np.sum(
            (fronting_isld[mask], first_lay_isld[mask], backing_isld[mask]),
            axis=0,
        ),
        np.sum(
            (
                np.zeros_like(fronting_msld)[mask],
                np.zeros_like(first_lay_msld)[mask],
                np.zeros_like(backing_msld)[mask],
            ),
            axis=0,
        ),
    )
)

z_and_slds_notreduced_expected_result = np.vstack(
    (
        z,
        np.sum((fronting_sld, first_lay_sld, backing_sld), axis=0),
        np.sum((fronting_isld, first_lay_isld, backing_isld), axis=0),
        np.sum((fronting_msld, first_lay_msld, backing_msld), axis=0),
    )
)

z_and_slds_notreduced_nospin_expected_result = np.vstack(
    (
        z,
        np.sum((fronting_sld, first_lay_sld, backing_sld), axis=0),
        np.sum((fronting_isld, first_lay_isld, backing_isld), axis=0),
        np.sum(
            (
                np.zeros_like(fronting_msld),
                np.zeros_like(first_lay_msld),
                np.zeros_like(backing_msld),
            ),
            axis=0,
        ),
    )
)

standard_examples_z_and_sld = [
    (
        nslds,
        thicknesses,
        roughnesses,
        "front",
        (0, 0, 0),
        isld,
        "none",
        False,
        z_and_slds_notreduced_nospin_expected_result,
    ),
    (
        nslds,
        thicknesses,
        roughnesses,
        "front",
        (0, 0, 0),
        isld,
        "none",
        True,
        z_and_slds_reduced_nospin_expected_result,
    ),
    (
        nslds,
        thicknesses,
        roughnesses,
        "front",
        msld,
        isld,
        "up",
        False,
        z_and_slds_notreduced_expected_result,
    ),
    (
        nslds,
        thicknesses,
        roughnesses,
        "front",
        msld,
        isld,
        "up",
        True,
        z_and_slds_reduced_expected_result,
    ),
    (
        nslds,
        thicknesses,
        roughnesses,
        "front",
        msld,
        isld,
        "down",
        False,
        z_and_slds_notreduced_expected_result,
    ),
    (
        nslds,
        thicknesses,
        roughnesses,
        "front",
        msld,
        isld,
        "down",
        True,
        z_and_slds_reduced_expected_result,
    ),
]


@pytest.mark.parametrize(
    "nslds, thicknesses, roughnesses, orientation, mslds, islds, spin_state,"
    "reduced, expected_result",
    standard_examples_z_and_sld,
)
def test_zs_and_slds(  # noqa : PLR0913
    nslds: tuple,
    thicknesses: tuple,
    roughnesses: tuple,
    orientation: str,
    mslds: tuple,
    islds: tuple,
    spin_state: str,
    reduced: bool,
    expected_result: float,
):
    vfp_kwargs = locals()
    del vfp_kwargs["expected_result"]
    del vfp_kwargs["reduced"]

    vfp = VFP(**vfp_kwargs)
    refnx_vfp = refnxVFP(**vfp_kwargs)
    refl1d_vfp = refl1dVFP(**vfp_kwargs)

    np.testing.assert_allclose(
        vfp.z_and_sld(reduced=reduced)[1],
        refnx_vfp.z_and_sld(reduced=reduced)[1],
    )
    np.testing.assert_allclose(
        vfp.z_and_sld(reduced=reduced)[1],
        refl1d_vfp.z_and_sld(reduced=reduced)[1],
    )
    np.testing.assert_allclose(
        refnx_vfp.z_and_sld(reduced=reduced)[1],
        refl1d_vfp.z_and_sld(reduced=reduced)[1],
    )

    # check vfp attr is the same expected result
    for v in [vfp, refnx_vfp, refl1d_vfp]:
        np.testing.assert_allclose(
            v.z_and_sld(reduced=reduced)[0], expected_result[0], atol=EPS
        )
        np.testing.assert_allclose(
            v.z_and_sld(reduced=reduced)[1], expected_result[1:].T, atol=EPS
        )


# def test_mslds():

# def test_islds():

# explicit BaseVFP methods
# -5 - (4 * 3) = -17
offset_expected_result_front = -17
# thicknesses = (0, 20), roughnesses = (3, 2) 20 + 12 + 5 = 37
# 37 + 4.5 (last dz when back is 4.5): -(41.5 - 20) = -21.5
offset_expected_result_back = -21.5
standard_examples_offset = [
    (
        nslds,
        thicknesses,
        roughnesses,
        "front",
        (0, 0, 0),
        isld,
        "none",
        offset_expected_result_front,
    ),
    (
        nslds,
        thicknesses,
        roughnesses,
        "back",
        msld,
        isld,
        "up",
        offset_expected_result_back,
    ),
]


@pytest.mark.parametrize(
    "nslds, thicknesses, roughnesses, orientation, mslds, islds, spin_state"
    ", expected_result",
    standard_examples_offset,
)
def test_sld_offset(  # noqa : PLR0913
    nslds: tuple,
    thicknesses: tuple,
    roughnesses: tuple,
    orientation: str,
    mslds: tuple,
    islds: tuple,
    spin_state: str,
    expected_result: np.ndarray,
):
    vfp_kwargs = locals()
    del vfp_kwargs["expected_result"]

    vfp = VFP(**vfp_kwargs)
    refnx_vfp = refnxVFP(**vfp_kwargs)
    refl1d_vfp = refl1dVFP(**vfp_kwargs)

    assert_allclose = partial(
        np.testing.assert_allclose, desired=expected_result, atol=EPS
    )

    map(
        assert_allclose,
        [vfp.sld_offset(), refnx_vfp.sld_offset(), refl1d_vfp.sld_offset()],
    )

    for v in [vfp, refnx_vfp, refl1d_vfp]:
        np.testing.assert_allclose(v.sld_offset(), expected_result, atol=EPS)


tuple_pars_expected_result = (
    (0, 20, 30),  # thicks
    (3, 2, 5),  # roughs
    (0, 1, 0, 0),  # mslds
    (3, 4),  # demag_widths
    (15, 25),  # demag_locs
)
standard_examples_tuple_pars = [
    (
        [2.07, 3.47, 0.2, 6.7],
        [0, 20, 30],
        [3, 2, 5],
        "front",
        [0, 1, 0, 0],
        [3e-6, 2e-6, 1e-6, 0],
        "up",
        [15, 25],
        [3, 4],
        tuple_pars_expected_result,
    ),
    (
        [2.07, 3.47, 0.2, 6.7],
        [0, 20, 30],
        [3, 2, 5],
        "front",
        [0, 1, 0, 0],
        [3e-6, 2e-6, 1e-6, 0],
        "down",
        [15, 25],
        [3, 4],
        tuple_pars_expected_result,
    ),
    (
        [2.07, 3.47, 0.2, 6.7],
        [0, 20, 30],
        [3, 2, 5],
        "back",
        [0, 1, 0, 0],
        [3e-6, 2e-6, 1e-6, 0],
        "up",
        [15, 25],
        [3, 4],
        tuple_pars_expected_result,
    ),
    (
        [2.07, 3.47, 0.2, 6.7],
        [0, 20, 30],
        [3, 2, 5],
        "back",
        [0, 1, 0, 0],
        [3e-6, 2e-6, 1e-6, 0],
        "down",
        [15, 25],
        [3, 4],
        tuple_pars_expected_result,
    ),
]


@pytest.mark.parametrize(
    "nslds, thicknesses, roughnesses, orientation, mslds, islds, spin_state"
    ", demaglocs, demagwidths, expected_result",
    standard_examples_tuple_pars,
)
def test_tuple_pars(  # noqa : PLR0913
    nslds: list,
    thicknesses: list,
    roughnesses: list,
    orientation: str,
    mslds: list,
    islds: list,
    spin_state: str,
    demaglocs: list,
    demagwidths: list,
    expected_result: tuple,
):
    vfp_kwargs = locals()
    del vfp_kwargs["expected_result"]

    vfp = VFP(**vfp_kwargs)
    refnx_vfp = refnxVFP(**vfp_kwargs)
    refl1d_vfp = refl1dVFP(**vfp_kwargs)

    vfp._tuple_pars()
    refnx_vfp._tuple_pars()
    refl1d_vfp._tuple_pars()

    assert_allclose = partial(
        np.testing.assert_allclose, desired=expected_result, atol=EPS
    )

    map(
        assert_allclose,
        [vfp.tup_thicks, refnx_vfp.tup_thicks, refl1d_vfp.tup_thicks],
    )

    map(
        assert_allclose,
        [vfp.tup_roughs, refnx_vfp.tup_roughs, refl1d_vfp.tup_roughs],
    )

    map(
        assert_allclose,
        [vfp.tup_mslds, refnx_vfp.tup_mslds, refl1d_vfp.tup_mslds],
    )

    map(
        assert_allclose,
        [
            vfp.tup_demag_widths,
            refnx_vfp.tup_demag_widths,
            refl1d_vfp.tup_demag_widths,
        ],
    )

    map(
        assert_allclose,
        [
            vfp.tup_demag_locs,
            refnx_vfp.tup_demag_locs,
            refl1d_vfp.tup_demag_locs,
        ],
    )

    for v in [vfp, refnx_vfp, refl1d_vfp]:
        np.testing.assert_allclose(v.tup_thicks, expected_result[0], atol=EPS)
        np.testing.assert_allclose(v.tup_roughs, expected_result[1], atol=EPS)
        np.testing.assert_allclose(v.tup_mslds, expected_result[2], atol=EPS)
        np.testing.assert_allclose(
            v.tup_demag_widths, expected_result[3], atol=EPS
        )
        np.testing.assert_allclose(
            v.tup_demag_locs, expected_result[4], atol=EPS
        )


arrtotuple_standard_examples = (
    (
        [2.07, 3.47, 0.2, 6.7],
        [0, 20, 30],
        [3, 2, 5],
        np.array([0, 20, 30]),
        (0, 20, 30),
    ),
    (
        [2.07, 3.47, 0.2, 6.7],
        [0, 20, 30],
        [3, 2, 5],
        np.array(
            [[2.07, 3.47, 0.2, 6.7], [0, 1, 0, 0], [3e-6, 2e-6, 1e-6, 0]]
        ),
        ((2.07, 3.47, 0.2, 6.7), (0, 1, 0, 0), (3e-6, 2e-6, 1e-6, 0)),
    ),
)


@pytest.mark.parametrize(
    "nslds, thicknesses, roughnesses, tuple_input, expected_result",
    arrtotuple_standard_examples,
)
def test_arrtotuple(
    nslds: list,
    thicknesses: list,
    roughnesses: list,
    tuple_input,
    expected_result: tuple,
):
    vfp_kwargs = locals()
    del vfp_kwargs["expected_result"]
    del vfp_kwargs["tuple_input"]

    vfp = VFP(**vfp_kwargs)
    refnx_vfp = refnxVFP(**vfp_kwargs)
    refl1d_vfp = refl1dVFP(**vfp_kwargs)

    vfp_tup = vfp._arrtotuple(tuple_input)
    refnx_tup = refnx_vfp._arrtotuple(tuple_input)
    refl1d_tup = refl1d_vfp._arrtotuple(tuple_input)

    assert_allclose = partial(
        np.testing.assert_allclose, desired=expected_result, atol=EPS
    )

    map(
        assert_allclose,
        [vfp_tup, refnx_tup, refl1d_tup],
    )
