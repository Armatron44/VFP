import numpy as np
from numpy.testing import assert_allclose
import scipy

from vfp.calc import (
    consecutive,
    get_demag,
    calc_dzs,
    calc_zeds,
    init_demag,
    calc_vfp,
    one_minus_cdf,
    integrate_vfp,
)

def init_standard_sample():
    # set up some quick standard test parameters.
    # 4 layers + use some non-integer values.
    lot = [0, 19.7, 50, 30]
    lor = [3.1, 5, 7.3, 6]
    nSLDs = [0, 6, 4, 3.47, 2.07]
    mSLDs = [0, 0, 3, 0, 0]
    locs = [1, 25]
    widths = [1, 5]
    conformal = [0, 0, 0, 0]

    dict_res = {
        "thicks": lot,
        "roughs": lor,
        "nSLDs": nSLDs,
        "mSLDs": mSLDs,
        "locs": locs,
        "widths": widths,
        "conformal": conformal,
    }

    return dict_res

def test_consecutive():
    arr_test = np.array([1, 2, 3, 5, 6, 7])
    consec_list = consecutive(arr_test)
    expected_output = [np.array([1, 2, 3]), np.array([5, 6, 7])]
    assert_allclose(consec_list, expected_output)

def test_calc_dzs():
    dz = calc_dzs(
        zstart=-17.5,
        zend=134,
        points=304,
        idxs=(10, 11, 12, 13, 14, 50, 51, 52, 53),
    )
    expected_output = np.ones(304 - 9) * 0.5
    # (15 - 9) * 0.5 = 3
    expected_output[10] = 3
    # (54 - 49) * 0.5 = 2.5
    expected_output[45] = 2.5
    assert_allclose(dz, expected_output)

def test_one_minus_cdf():
    sample_dict = init_standard_sample()

    z = np.linspace(-17.5, 134, 304)
    first_interface = 1 - scipy.stats.norm.cdf(
        x=z, loc=sample_dict["thicks"][0], scale=sample_dict["roughs"][0]
    )
    second_interface = 1 - scipy.stats.norm.cdf(
        x=z,
        loc=sample_dict["thicks"][0] + sample_dict["thicks"][1],
        scale=sample_dict["roughs"][1],
    )
    third_interface = 1 - scipy.stats.norm.cdf(
        x=z,
        loc=sample_dict["thicks"][0]
        + sample_dict["thicks"][1]
        + sample_dict["thicks"][2],
        scale=sample_dict["roughs"][2],
    )
    fourth_interface = 1 - scipy.stats.norm.cdf(
        x=z,
        loc=sample_dict["thicks"][0]
        + sample_dict["thicks"][1]
        + sample_dict["thicks"][2]
        + sample_dict["thicks"][3],
        scale=sample_dict["roughs"][3],
    )

    expected_output = np.vstack(
        (first_interface, second_interface, third_interface, fourth_interface)
    )

    tup_thicks = tuple(sample_dict["thicks"])
    arr_thicks = np.array(tup_thicks)
    cumthick = np.cumsum(arr_thicks)

    tup_roughs = tuple(sample_dict["roughs"])
    arr_roughs = np.array(tup_roughs)

    first_interf_output = one_minus_cdf(z, cumthick[0], arr_roughs[0])
    second_interf_output = one_minus_cdf(z, cumthick[1], arr_roughs[1])
    third_interf_output = one_minus_cdf(z, cumthick[2], arr_roughs[2])
    fourth_interf_output = one_minus_cdf(z, cumthick[3], arr_roughs[3])

    real_output = np.vstack(
        (
            first_interf_output,
            second_interf_output,
            third_interf_output,
            fourth_interf_output,
        )
    )

    assert_allclose(real_output, expected_output)

def test_init_demag():
    sample_dict = init_standard_sample()

    z = np.linspace(-17.5, 134, 304)
    first_layer_vfp = scipy.stats.norm.cdf(
        x=z, loc=sample_dict["thicks"][0], scale=sample_dict["roughs"][0]
    )
    second_layer_vfp = scipy.stats.norm.cdf(
        x=z, loc=sample_dict["thicks"][1], scale=sample_dict["roughs"][1]
    )
    third_layer_vfp = scipy.stats.norm.cdf(
        x=z, loc=sample_dict["thicks"][2], scale=sample_dict["roughs"][2]
    )
    fourth_layer_vfp = scipy.stats.norm.cdf(
        x=z, loc=sample_dict["thicks"][3], scale=sample_dict["roughs"][3]
    )
    expected_vfp = np.vstack(
        (
            1 - first_layer_vfp,
            first_layer_vfp * (1 - second_layer_vfp),
            first_layer_vfp * second_layer_vfp * (1 - third_layer_vfp),
            first_layer_vfp
            * second_layer_vfp
            * third_layer_vfp
            * (1 - fourth_layer_vfp),
            first_layer_vfp
            * second_layer_vfp
            * third_layer_vfp
            * fourth_layer_vfp,
        )
    )
    res = init_demag(
        locs=tuple(sample_dict["locs"]),
        widths=tuple(sample_dict["widths"]),
        mSLDs=tuple(sample_dict["mSLDs"]),
        zeds=tuple(np.linspace(-17.5, 134, 304)),
        vfp=tuple(tuple(i) for i in expected_vfp),
    )

    expected_demag_arr = np.ones(shape=(len(sample_dict["mSLDs"]), 304))
    expected_demag_arr[2] = 1 - (
        scipy.stats.norm.cdf(np.linspace(-17.5, 134, 304), loc=1, scale=1)
        * (
            1
            - scipy.stats.norm.cdf(
                np.linspace(-17.5, 134, 304), loc=1 + 25, scale=5
            )
        )
    )
    exp_mag_comp = expected_vfp * expected_demag_arr
    difference_arr = np.abs(np.diff(exp_mag_comp, axis=1)) < 1e-5
    reduce_diff_arr = np.all(difference_arr, axis=0)
    indices_full = np.nonzero(reduce_diff_arr)

    # shift indices along by 1 & don't take last value of indices_full.
    expected_idx = (indices_full[0] + 1)[:-1]

    # now remove parts of the vfps and mag_comp where they are ~ invariant.
    reduced_vfp = np.delete(expected_vfp, expected_idx, 1)
    reduced_magcomp = np.delete(exp_mag_comp, expected_idx, 1)

    assert_allclose(res[0], reduced_vfp)
    assert_allclose(res[1], reduced_magcomp)
    assert_allclose(res[2], expected_idx)
    assert_allclose(res[3], expected_demag_arr)

def test_calc_zeds():
    sample_dict = init_standard_sample()
    zeds = calc_zeds(
        rough=tuple(sample_dict["roughs"]),
        thick=tuple(sample_dict["thicks"]),
        mxdz=0.5,
    )
    expected_output = np.linspace(-17.5, 134, 304)
    assert_allclose(zeds, expected_output)

def test_calc_vfp():
    sample_dict = init_standard_sample()

    vfp_res = calc_vfp(
        rough=tuple(sample_dict["roughs"]),
        thick=tuple(sample_dict["thicks"]),
        zeds=tuple(np.linspace(-17.5, 134, 304)),
        conformal=tuple(sample_dict["conformal"]),
    )

    z = np.linspace(-17.5, 134, 304)
    first_interface = scipy.stats.norm.cdf(
        x=z, loc=sample_dict["thicks"][0], scale=sample_dict["roughs"][0]
    )
    second_interface = scipy.stats.norm.cdf(
        x=z,
        loc=sample_dict["thicks"][0] + sample_dict["thicks"][1],
        scale=sample_dict["roughs"][1],
    )
    third_interface = scipy.stats.norm.cdf(
        x=z,
        loc=sample_dict["thicks"][0]
        + sample_dict["thicks"][1]
        + sample_dict["thicks"][2],
        scale=sample_dict["roughs"][2],
    )
    fourth_interface = scipy.stats.norm.cdf(
        x=z,
        loc=sample_dict["thicks"][0]
        + sample_dict["thicks"][1]
        + sample_dict["thicks"][2]
        + sample_dict["thicks"][3],
        scale=sample_dict["roughs"][3],
    )

    expected_output = np.vstack(
        (
            1 - first_interface,
            first_interface * (1 - second_interface),
            first_interface * second_interface * (1 - third_interface),
            first_interface
            * second_interface
            * third_interface
            * (1 - fourth_interface),
            first_interface
            * second_interface
            * third_interface
            * fourth_interface,
        )
    )

    assert_allclose(vfp_res, expected_output, atol=np.finfo(float).eps, rtol=0)

def test_get_demag():
    zed = np.linspace(-17.5, 134, 304)
    # test no width and no locs case
    expected_output = np.zeros_like(zed)
    real_output = get_demag(dist=zed, locs=np.array([]), widths=np.array([]))
    assert_allclose(real_output, expected_output)
    # test 1 set of widths and locs
    real_output = get_demag(
        dist=zed, locs=np.array([1, 25]), widths=np.array([1, 5])
    )
    expected_output = (
        scipy.stats.norm.cdf(zed, loc=1, scale=1)
        * (1 - scipy.stats.norm.cdf(zed, loc=1 + 25, scale=5))
    )
    assert_allclose(real_output, expected_output)
    # test 2 set of widths and locs.
    real_output = get_demag(
        dist=zed, locs=np.array([1, 25, 1, 60]), widths=np.array([1, 5, 3, 6])
    )
    cumlocs = np.cumsum(np.array([1, 25, 1, 60]))
    peak1 = scipy.stats.norm.cdf(zed, loc=cumlocs[0], scale=1) * (
        1 - scipy.stats.norm.cdf(zed, loc=cumlocs[1], scale=5)
    )
    peak2 = (
        scipy.stats.norm.cdf(zed, loc=cumlocs[1], scale=5)
        * scipy.stats.norm.cdf(zed, loc=cumlocs[2], scale=3)
        * (1 - scipy.stats.norm.cdf(zed, loc=cumlocs[3], scale=6))
    )

    expected_output = peak1 + peak2
    assert_allclose(real_output, expected_output)

def test_integrate_vfp():
    zed = np.linspace(-10, 10, 10001)
    first_peak = scipy.stats.norm.pdf(zed, loc=0, scale=1)
    second_peak = 4 * scipy.stats.norm.pdf(zed, loc=0, scale=1)
    vfps = np.vstack((first_peak, second_peak))
    vfps = tuple(tuple(i) for i in vfps)
    first_res, second_res = integrate_vfp(
        zeds=tuple(zed),
        indexs=(),
        red_vfps=vfps,
        layer_indices=tuple([0, 1])
    )
    first_expected, second_expected = (1, 4)
    assert_allclose(first_res, first_expected)
    assert_allclose(second_res, second_expected)
