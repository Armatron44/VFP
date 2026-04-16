from contextlib import nullcontext

import numpy as np
import pytest
import scipy
from numpy.testing import assert_allclose

from vfp.calc import (
    arr_to_tuple,
    calc_demag_array,
    calc_dzs,
    calc_indices,
    calc_vfp,
    calc_zeds,
    consecutive,
    get_demag,
    heaviside_step,
    integrate_vfp,
    one_minus_cdf,
    reduce_vfp_and_magcomp,
    transform_indices,
)


@pytest.mark.parametrize(
    "array, expected_result",
    [
        pytest.param(
            np.array([1, 2, 3, 5, 6, 7]),
            nullcontext([np.array([1, 2, 3]), np.array([5, 6, 7])]),
            id="First consecutive test.",
        ),
        pytest.param(
            np.array([100, 101, 105, 106, 200, 201]),
            nullcontext(
                [
                    np.array([100, 101]),
                    np.array([105, 106]),
                    np.array([200, 201]),
                ]
            ),
            id="Second consecutive test.",
        ),
        pytest.param(
            np.random.default_rng().normal(size=(3, 100)),
            pytest.raises(ValueError),
            id="Third consecutive test, ValueError.",
        ),
        pytest.param(
            np.random.default_rng().integers(low=1, high=10, size=1).item(),
            pytest.raises(TypeError),
            id="Fourth consecutive test, TypeError.",
        ),
    ],
)
def test_consecutive(array, expected_result):
    with expected_result as e:
        consec_list = consecutive(array)
        assert_allclose(consec_list, e)


calc_dz_eo_first = np.ones(303 - 7) * 0.5
# 5 * 0.5 = 2.5
calc_dz_eo_first[10] = 2.5
# 4 * 0.5 = 2
calc_dz_eo_first[46] = 2
calc_dz_eo_second = np.ones(20) * 0.5


@pytest.mark.parametrize(
    "zstart, zend, points, idxs, expected_result",
    [
        pytest.param(
            -17.5,
            134,
            304,
            (10, 11, 12, 13, 14, 50, 51, 52, 53),
            calc_dz_eo_first,
            id="First calc_dz test, two batches of consec idxs.",
        ),
        pytest.param(
            0,
            10,
            21,
            (),
            calc_dz_eo_second,
            id="Second calc_dz test, no idxs.",
        ),
    ],
)
def test_calc_dzs(
    zstart: float,
    zend: float,
    points: int,
    idxs: tuple[int, ...],
    expected_result: np.ndarray,
) -> None:
    dz = calc_dzs(
        zstart=zstart,
        zend=zend,
        points=points,
        idxs=idxs,
    )
    assert_allclose(dz, expected_result)


z = np.linspace(-17.5, 134, 304)
first_interface = 1 - scipy.stats.norm.cdf(x=z, loc=0, scale=3.1)
second_interface = 1 - scipy.stats.norm.cdf(
    x=z,
    loc=0 + 19.7,
    scale=5,
)
third_interface = 1 - scipy.stats.norm.cdf(
    x=z,
    loc=0 + 19.7 + 50,
    scale=7.3,
)
fourth_interface = 1 - scipy.stats.norm.cdf(
    x=z,
    loc=0 + 19.7 + 50 + 30,
    scale=6,
)


@pytest.mark.parametrize(
    "z, idx, expected_output",
    [
        pytest.param(z, 0, first_interface, id="First one_minus_cdf test."),
        pytest.param(z, 1, second_interface, id="Second one_minus_cdf test."),
        pytest.param(z, 2, third_interface, id="Third one_minus_cdf test."),
        pytest.param(z, 3, fourth_interface, id="Fourth one_minus_cdf test."),
    ],
)
def test_one_minus_cdf(
    z: np.ndarray, idx: int, expected_output: np.ndarray, init_standard_sample
) -> None:
    sample_dict = init_standard_sample
    # follow same process as in vfp
    tup_thicks = tuple(sample_dict["thicks"])
    arr_thicks = np.array(tup_thicks)
    cumthick = np.cumsum(arr_thicks)
    tup_roughs = tuple(sample_dict["roughs"])
    arr_roughs = np.array(tup_roughs)
    output = one_minus_cdf(z, cumthick[idx], arr_roughs[idx])
    assert_allclose(output, expected_output)


first_demag_arr_expected = np.ones((5, 304))
first_demag_factor = scipy.stats.norm.cdf(
    np.linspace(-17.5, 134, 304), loc=1, scale=1
) * (
    1
    - scipy.stats.norm.cdf(np.linspace(-17.5, 134, 304), loc=1 + 25, scale=5)
)
first_demag_arr_expected[2] = 1 - first_demag_factor

second_demag_arr_expected = np.ones((5, 258))
second_demag_factor = scipy.stats.norm.cdf(
    np.linspace(-21.5, 107, 258), loc=1, scale=1.3
) * (
    1
    - scipy.stats.norm.cdf(np.linspace(-21.5, 107, 258), loc=1 + 23, scale=7)
) + scipy.stats.norm.cdf(
    np.linspace(-21.5, 107, 258), loc=1 + 23, scale=7
) * scipy.stats.norm.cdf(
    np.linspace(-21.5, 107, 258), loc=1 + 23 + 18, scale=4.2
) * (
    1
    - scipy.stats.norm.cdf(
        np.linspace(-21.5, 107, 258), loc=1 + 23 + 18 + 22, scale=5.1
    )
)
second_demag_arr_expected[1] = 1 - second_demag_factor
second_demag_arr_expected[3] = 1 - second_demag_factor


@pytest.mark.parametrize(
    "zed, standard_sample_name, expected_result",
    [
        pytest.param(
            np.linspace(-17.5, 134, 304),
            "init_standard_sample",
            first_demag_arr_expected,
            id="First calc_demag_arr test.",
        ),
        pytest.param(
            np.linspace(-21.5, 107, 258),
            "init_standard_sample_two",
            second_demag_arr_expected,
            id="Second calc_demag_arr test.",
        ),
    ],
)
def test_calc_demag_array(
    zed: np.ndarray,
    standard_sample_name: str,
    expected_result: np.ndarray,
    request: pytest.FixtureRequest,
) -> None:
    sample_dict = request.getfixturevalue(standard_sample_name)
    locs = sample_dict["locs"]
    widths = sample_dict["widths"]
    mslds = sample_dict["mslds"]
    tup_locs, tup_widths, tup_mslds = tuple(locs), tuple(widths), tuple(mslds)
    output = calc_demag_array(tup_locs, tup_widths, tup_mslds, tuple(zed))
    assert_allclose(output, expected_result, atol=np.finfo(float).eps)


@pytest.mark.parametrize(
    "standard_sample_name, mxdz, expected_result",
    [
        pytest.param(
            "init_standard_sample",
            0.5,
            np.linspace(-17.5, 134, 304),
            id="First calc_dz test.",
        ),
        pytest.param(
            "init_standard_sample_two",
            1,
            np.linspace(-22, 107, 130),
            id="Second calc_dz test.",
        ),
    ],
)
def test_calc_zeds(
    standard_sample_name: str,
    mxdz: float,
    expected_result: np.ndarray,
    request: pytest.FixtureRequest,
) -> None:
    sample_dict = request.getfixturevalue(standard_sample_name)
    zeds = calc_zeds(
        rough=tuple(sample_dict["roughs"]),
        thick=tuple(sample_dict["thicks"]),
        mxdz=mxdz,
    )
    assert_allclose(zeds, expected_result)


z1 = np.linspace(-17.5, 134, 304)
first_vfp_first_interface = scipy.stats.norm.cdf(x=z1, loc=0, scale=3.1)
first_vfp_second_interface = scipy.stats.norm.cdf(
    x=z1,
    loc=0 + 19.7,
    scale=5,
)
first_vfp_third_interface = scipy.stats.norm.cdf(
    x=z1,
    loc=0 + 19.7 + 50,
    scale=7.3,
)
first_vfp_fourth_interface = scipy.stats.norm.cdf(
    x=z1,
    loc=0 + 19.7 + 50 + 30,
    scale=6,
)

first_vfp_expected_output = np.vstack(
    (
        1 - first_vfp_first_interface,
        first_vfp_first_interface * (1 - first_vfp_second_interface),
        first_vfp_first_interface
        * first_vfp_second_interface
        * (1 - first_vfp_third_interface),
        first_vfp_first_interface
        * first_vfp_second_interface
        * first_vfp_third_interface
        * (1 - first_vfp_fourth_interface),
        first_vfp_first_interface
        * first_vfp_second_interface
        * first_vfp_third_interface
        * first_vfp_fourth_interface,
    )
)
z2 = np.linspace(-21.5, 107, 258)
second_vfp_first_interface = scipy.stats.norm.cdf(x=z2, loc=0, scale=4.1)
second_vfp_second_interface = scipy.stats.norm.cdf(
    x=z2,
    loc=0 + 32.8,
    scale=4.1,
)
second_vfp_third_interface = scipy.stats.norm.cdf(
    x=z2,
    loc=0 + 32.8 + 16.3,
    scale=7,
)
second_vfp_fourth_interface = scipy.stats.norm.cdf(
    x=z2,
    loc=0 + 32.8 + 16.3 + 24.6,
    scale=3,
)

second_vfp_expected_output = np.vstack(
    (
        1 - second_vfp_first_interface,  # fronting
        (1 - second_vfp_second_interface)
        - (1 - second_vfp_first_interface),  # lay1
        second_vfp_second_interface
        * (1 - second_vfp_third_interface),  # lay2
        second_vfp_second_interface
        * second_vfp_third_interface
        * (1 - second_vfp_fourth_interface),  # lay3
        1
        - np.sum(  # backing
            (
                1 - second_vfp_first_interface,  # f
                (1 - second_vfp_second_interface)
                - (1 - second_vfp_first_interface),  # lay1
                second_vfp_second_interface
                * (1 - second_vfp_third_interface),  # lay2
                second_vfp_second_interface
                * second_vfp_third_interface
                * (1 - second_vfp_fourth_interface),  # lay3
            ),
            axis=0,
        ),
    )
)


@pytest.mark.parametrize(
    "zeds, expected_result, standard_sample_name",
    [
        pytest.param(
            np.linspace(-17.5, 134, 304),
            first_vfp_expected_output,
            "init_standard_sample",
            id="First calc_vfp test.",
        ),
        pytest.param(
            np.linspace(-21.5, 107, 258),
            second_vfp_expected_output,
            "init_standard_sample_two",
            id="Second calc_vfp test.",
        ),
    ],
)
def test_calc_vfp(
    zeds: np.ndarray,
    expected_result: np.ndarray,
    standard_sample_name: str,
    request: pytest.FixtureRequest,
) -> None:
    sample_dict = request.getfixturevalue(standard_sample_name)
    thicks, roughs, conformals = (
        sample_dict["thicks"],
        sample_dict["roughs"],
        sample_dict["conformal"],
    )

    vfp_res = calc_vfp(
        rough=tuple(roughs),
        thick=tuple(thicks),
        zeds=tuple(zeds),
        conformal=tuple(conformals),
    )
    assert_allclose(vfp_res, expected_result, atol=np.finfo(float).eps)


first_test_demag_res = scipy.stats.norm.cdf(
    np.linspace(-17.5, 134, 304), loc=1, scale=1
) * (
    1
    - scipy.stats.norm.cdf(np.linspace(-17.5, 134, 304), loc=1 + 25, scale=5)
)

second_test_demag_res = scipy.stats.norm.cdf(
    np.linspace(-21.5, 107, 258), loc=1, scale=1.3
) * (
    1
    - scipy.stats.norm.cdf(np.linspace(-21.5, 107, 258), loc=1 + 23, scale=7)
) + scipy.stats.norm.cdf(
    np.linspace(-21.5, 107, 258), loc=1 + 23, scale=7
) * scipy.stats.norm.cdf(
    np.linspace(-21.5, 107, 258), loc=1 + 23 + 18, scale=4.2
) * (
    1
    - scipy.stats.norm.cdf(
        np.linspace(-21.5, 107, 258), loc=1 + 23 + 18 + 22, scale=5.1
    )
)


@pytest.mark.parametrize(
    "zed, standard_sample_name, expected_result",
    [
        pytest.param(
            np.linspace(-17.5, 134, 304),
            "init_standard_sample",
            first_test_demag_res,
            id="First get_demag test.",
        ),
        pytest.param(
            np.linspace(-21.5, 107, 258),
            "init_standard_sample_two",
            second_test_demag_res,
            id="Two get_demag test.",
        ),
    ],
)
def test_get_demag(
    zed: np.ndarray,
    standard_sample_name: str,
    expected_result: np.ndarray,
    request: pytest.FixtureRequest,
) -> None:
    sample_dict = request.getfixturevalue(standard_sample_name)
    locs, widths = sample_dict["locs"], sample_dict["widths"]
    real_output = get_demag(zed, locs, widths)
    assert_allclose(real_output, expected_result)


@pytest.mark.parametrize(
    "zed, peaks, indices, layer_indices, expected_result",
    [
        pytest.param(
            np.linspace(-10, 10, 10001),
            (
                scipy.stats.skewnorm.pdf(
                    np.linspace(-10, 10, 10001), a=4, loc=0, scale=1
                ),
            ),
            (),
            (0,),
            nullcontext((1,)),
            id="First test_integrate_test.",
        ),
        pytest.param(
            np.linspace(-10, 10, 10001),
            (
                scipy.stats.norm.pdf(
                    np.delete(np.linspace(-10, 10, 10001), np.array([5, 6])),
                    loc=0,
                    scale=1,
                ),
                4
                * scipy.stats.norm.pdf(
                    np.delete(np.linspace(-10, 10, 10001), np.array([5, 6])),
                    loc=0,
                    scale=1,
                ),
            ),
            (4, 5, 6),
            (0, 1),
            nullcontext((1, 4)),
            id="Second test_integrate_test.",
        ),
        pytest.param(
            np.linspace(-10, 10, 10001),
            (
                scipy.stats.norm.pdf(
                    np.linspace(-10, 10, 10001), loc=0, scale=1
                ),
                4
                * scipy.stats.norm.pdf(
                    np.linspace(-10, 10, 10001), loc=0, scale=1
                ),
            ),
            (),
            (),
            pytest.raises(ValueError),
            id="Third test_integrate_test.",
        ),
    ],
)
def test_integrate_vfp(
    zed: np.ndarray,
    peaks: tuple[np.ndarray, ...],
    indices: tuple[int, ...],
    layer_indices: tuple[int, ...],
    expected_result: tuple[float, ...],
) -> None:
    vfps = np.vstack(peaks)
    vfps = tuple(tuple(i) for i in vfps)
    with expected_result as e:
        res_list = integrate_vfp(
            zeds=tuple(zed),
            indexs=indices,
            red_vfps=vfps,
            layer_indices=layer_indices,
        )
        for res, expec in zip(res_list, e, strict=False):
            assert_allclose(res, expec)


expected_result_heaviside_1 = np.zeros_like(np.linspace(-10, 10, 201))
expected_result_heaviside_1[130:] = 1
expected_result_heaviside_2 = np.zeros_like(np.linspace(-10, 10, 201))
expected_result_heaviside_2[50:] = 1
standard_examples_heaviside = [
    (np.linspace(-10, 10, 201), 3, expected_result_heaviside_1),
    (np.linspace(-10, 10, 201), -5, expected_result_heaviside_2),
]


@pytest.mark.parametrize(
    "z, loc, expected_result",
    standard_examples_heaviside,
)
def test_heaviside_step(
    z: np.ndarray, loc: int, expected_result: np.ndarray
) -> None:
    step_fn = heaviside_step(z, loc=loc)
    assert_allclose(step_fn, expected_result)


@pytest.mark.parametrize(
    "raw_idx, expected_result",
    [
        pytest.param(
            np.array([1, 2, 3, 4, 5, 6]),
            np.array([2, 3, 4, 5, 6]),
            id="First transform_indices test.",
        ),
        pytest.param(
            np.array([1, 2, 3, 4, 12, 13, 14, 15, 20]),
            np.array([2, 3, 4, 13, 14, 15]),
            id="Second transform_indices test.",
        ),
    ],
)
def test_transform_indices(
    raw_idx: np.ndarray, expected_result: np.ndarray
) -> None:
    output = transform_indices(raw_idx)
    assert_allclose(output, expected_result)


vfp_f = np.concatenate([np.ones(20), np.linspace(1, 0, 6), np.zeros(23)])
vfp_b = np.concatenate([np.zeros(20), np.linspace(0, 1, 6), np.ones(23)])
vfp_first = np.vstack([vfp_f, vfp_b])
vfp_first_tuple = tuple(tuple(r) for r in vfp_first)
demag_arr_first_tuple = tuple(tuple(r) for r in np.ones(shape=(2, 49)))
# There are 21 ones/zeros in a row at the start and
# last digit of linspace(1, 0, 6) is zero, so there are 23
# points that are equal at the end.
expected_res_first = np.concatenate([np.arange(20), np.arange(25, 25 + 23)])

vfp_f = np.concatenate([np.ones(5), np.linspace(1, 0, 9), np.zeros(19)])
vfp_lay1 = np.concatenate(
    [
        np.zeros(5),
        np.linspace(0, 0.75, 7),
        np.linspace(0.75, 0, 5),
        np.zeros(16),
    ]
)
vfp_b = 1 - (vfp_f + vfp_lay1)
vfp_second = np.vstack([vfp_f, vfp_lay1, vfp_b])
vfp_second_tuple = tuple(tuple(r) for r in vfp_second)
demag_arr_second_tuple = tuple(tuple(r) for r in np.ones(shape=(3, 33)))
# 5 zeros/ones at the start. Only have stable vfps after 5+7+5 - 1
# points.
expected_res_second = np.concatenate(
    [np.arange(5), np.arange(5 + 6 + 5, 5 + 6 + 5 + 16)]
)


@pytest.mark.parametrize(
    "vfp, demag_arr, expected_result",
    [
        pytest.param(
            vfp_first_tuple,
            demag_arr_first_tuple,
            expected_res_first,
            id="First calc_indices test.",
        ),
        pytest.param(
            vfp_second_tuple,
            demag_arr_second_tuple,
            expected_res_second,
            id="Second calc_indices test.",
        ),
    ],
)
def test_calc_indices(
    vfp: tuple[tuple[float, ...]],
    demag_arr: tuple[tuple[float, ...]],
    expected_result: np.ndarray,
) -> None:
    output = calc_indices(vfp, demag_arr)
    assert_allclose(output, expected_result)


# transformed indices are +1 leaving the last off in a consecutive series.
raw_idx_first = expected_res_first
expect_idx_first = np.concatenate(
    [np.arange(1, 20), np.arange(25 + 1, 25 + 23)]
)
expected_first_res = np.delete(vfp_first, expect_idx_first, 1)

raw_idx_second = expected_res_second
expect_idx_second = np.concatenate(
    [np.arange(1, 5), np.arange(5 + 6 + 5 + 1, 5 + 6 + 5 + 16)]
)
expected_second_res = np.delete(vfp_second, expect_idx_second, 1)


@pytest.mark.parametrize(
    "vfp, magcomp, idx, expected_result",
    [
        pytest.param(
            vfp_first,
            vfp_first * np.ones(shape=(2, 49)),
            raw_idx_first,
            expected_first_res,
            id="First reduce_vfp_and_magcomp test.",
        ),
        pytest.param(
            vfp_second,
            vfp_second * np.ones(shape=(3, 33)),
            raw_idx_second,
            expected_second_res,
            id="Second reduce_vfp_and_magcomp test.",
        ),
    ],
)
def test_reduce_vfp_and_magcomp(
    vfp: np.ndarray,
    magcomp: np.ndarray,
    idx: np.ndarray,
    expected_result: np.ndarray,
) -> None:
    reduced_vfp_output, reduced_magcomp_output = reduce_vfp_and_magcomp(
        vfp, magcomp, idx
    )
    assert_allclose(reduced_vfp_output, expected_result)
    # as this example has no deviation in demag_arr from 1,
    # magcomp will be equal to vfp.
    assert_allclose(reduced_magcomp_output, expected_result)


@pytest.mark.parametrize(
    "tuple_input, expected_result",
    [
        pytest.param(
            np.array([0, 20, 30]),
            nullcontext((0, 20, 30)),
            id="First arr_to_tuple test",
        ),
        pytest.param(
            np.array(
                [[2.07, 3.47, 0.2, 6.7], [0, 1, 0, 0], [3e-6, 2e-6, 1e-6, 0]]
            ),
            nullcontext(
                ((2.07, 3.47, 0.2, 6.7), (0, 1, 0, 0), (3e-6, 2e-6, 1e-6, 0))
            ),
            id="Second arr_to_tuple test",
        ),
        pytest.param(
            [5, 3, 2],
            pytest.raises(TypeError),
            id="Third arr_to_tuple test - not array.",
        ),
        pytest.param(
            np.array(
                [[[5, 3, 2], [1, 2, 4]], [[0.1, 1.4, 2.1], [1.1, 4.7, 5.5]]]
            ),
            nullcontext(
                (((5, 3, 2), (1, 2, 4)), ((0.1, 1.4, 2.1), (1.1, 4.7, 5.5)))
            ),
            id="Fourth arr_to_tuple test",
        ),
    ],
)
def test_arr_to_tuple(
    tuple_input,
    expected_result,
):
    with expected_result as e:
        out = arr_to_tuple(tuple_input)
        assert_allclose(out, desired=e)
