"""
Tests VFP and BaseVFP methods.
"""

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
start = -17  # -5 - (4 * 3) = -17
end = 37  # 5 + (0 + 20) + 4 * 4.5 = 37
z = np.linspace(start, end, int((end - start) / 0.5) + 1)
fronting_sld = (1 - norm.cdf(z, loc=0, scale=3)) * 0
fronting_msld = (1 - norm.cdf(z, loc=0, scale=3)) * 0
fronting_isld = (1 - norm.cdf(z, loc=0, scale=3)) * 0
first_lay_sld = 3 * (
    norm.cdf(z, loc=0, scale=3) * (1 - norm.cdf(z, loc=20, scale=2))
)
first_lay_msld = 0 * (
    norm.cdf(z, loc=0, scale=3) * (1 - norm.cdf(z, loc=20, scale=2))
)
first_lay_isld = 0 * (
    norm.cdf(z, loc=0, scale=3) * (1 - norm.cdf(z, loc=20, scale=2))
)
backing_sld = 4.5 * (
    norm.cdf(z, loc=0, scale=3) * norm.cdf(z, loc=20, scale=2)
)
backing_msld = 0 * (
    norm.cdf(z, loc=0, scale=3) * norm.cdf(z, loc=20, scale=2)
)
backing_isld = 0 * (
    norm.cdf(z, loc=0, scale=3) * norm.cdf(z, loc=20, scale=2)
)
standard_examples_orientation = [
    (
        (0, 3, 4.5),  # nslds
        (0, 20),  # thicknesses
        (3, 2),  # roughnesses
        "front",  # orientation
        (0, 0, 0),  # msld
        (0, 0, 0),  # isld
        "none",
        np.vstack(
            (
                z,
                np.sum((fronting_sld, first_lay_sld, backing_sld), axis=0),
                np.sum((fronting_msld, first_lay_msld, backing_msld), axis=0),
                np.sum((fronting_isld, first_lay_isld, backing_isld), axis=0),
            )
        ),
    ),
    (
        (0, 3, 4.5),  # nslds
        (0, 20),  # thicknesses
        (3, 2),  # roughnesses
        "back",  # orientation
        (0, 0, 0),  # msld
        (0, 0, 0),  # isld
        "none",
        np.vstack(
            (
                -(z - (0 + 20)),  # orientation = back - flip z.
                np.sum((fronting_sld, first_lay_sld, backing_sld), axis=0),
                np.sum((fronting_msld, first_lay_msld, backing_msld), axis=0),
                np.sum((fronting_isld, first_lay_isld, backing_isld), axis=0),
            )
        ),
    ),
]


@pytest.mark.parametrize(
    "slds, thicknesses, roughnesses, orientation, msld, isld, spin_state"
    ", expected_result",
    standard_examples_orientation,
)
def test_slds_orientation(  # noqa : PLR0913
    slds: tuple,
    thicknesses: tuple,
    roughnesses: tuple,
    orientation: str,
    msld: tuple,
    isld: tuple,
    spin_state: str,
    expected_result: np.ndarray,
):
    vfp = VFP(
        nslds=slds,
        thicknesses=thicknesses,
        roughnesses=roughnesses,
        orientation=orientation,
        mslds=msld,
        islds=isld,
        spin_state=spin_state,
    )
    refnx_vfp = refnxVFP(
        nslds=slds,
        thicknesses=thicknesses,
        roughnesses=roughnesses,
        orientation=orientation,
        mslds=msld,
        islds=isld,
        spin_state=spin_state,
    )
    refl1d_vfp = refl1dVFP(
        nslds=slds,
        thicknesses=thicknesses,
        roughnesses=roughnesses,
        orientation=orientation,
        mslds=msld,
        islds=isld,
        spin_state=spin_state,
    )

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


# def test_mslds():

# def test_islds():

# # explicit BaseVFP methods
# def test_sld_offset()

# def test_tuple_pars()

# def test_arrtotuple()

# def test_z_and_sld()
