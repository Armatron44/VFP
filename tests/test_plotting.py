import pathlib
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from numpy.testing import assert_allclose
from scipy import stats

from vfp import VFP, refnxVFP
from vfp.plotting import AxesIndex, PlotType, surfaces_for_display

EPS = np.finfo(float).eps

PLOTTING_EXAMPLES_PATH = (
    pathlib.Path(__file__).parent
    / "plotting_examples"
    / "plotting_examples.npz"
)

PLOTTING_EXAMPLES_MAP = {
    (
        "front",
        "allplots_layermats_mslice_tsld",
    ): "front_allplots_layermats_mslice_tsld",
    (
        "front",
        "allplots_layermats_mslice_align_at_2",
    ): "front_allplots_layermats_mslice_align_at_2",
    ("front", "sldvfpplots_layermats"): "front_sldvfpplots_layermats",
    ("front", "vfp_post"): "front_vfp_post",
    ("front", "surfacesvfp"): "front_surfacesvfp",
    ("front", "surfacesvfp_align_at_3"): "front_surfacesvfp_align_at_3",
    (
        "back_ssup",
        "allplots_layermats_mslice_tsld",
    ): "back_ssup_allplots_layermats_mslice_tsld",
    (
        "back_ssup",
        "allplots_layermats_mslice_align_at_2",
    ): "back_ssup_allplots_layermats_mslice_align_at_2",
    ("back_ssup", "sldvfpplots_layermats"): "back_ssup_sldvfpplots_layermats",
    ("back_ssup", "vfp_post"): "back_ssup_vfp_post",
    ("back_ssup", "surfacesvfp"): "back_ssup_surfacesvfp",
    (
        "back_ssup",
        "surfacesvfp_align_at_3",
    ): "back_ssup_surfacesvfp_align_at_3",
    (
        "front_mdz_01_ss_down",
        "allplots_layermats_mslice_tsld",
    ): "front_mdz_01_ss_down_allplots_layermats_mslice_tsld",
    (
        "front_mdz_01_ss_down",
        "allplots_layermats_mslice_align_at_2",
    ): "front_mdz_01_ss_down_allplots_layermats_mslice_align_at_2",
    (
        "front_mdz_01_ss_down",
        "sldvfpplots_layermats",
    ): "front_mdz_01_ss_down_sldvfpplots_layermats",
    ("front_mdz_01_ss_down", "vfp_post"): "front_mdz_01_ss_down_vfp_post",
    (
        "front_mdz_01_ss_down",
        "surfacesvfp",
    ): "front_mdz_01_ss_down_surfacesvfp",
    (
        "front_mdz_01_ss_down",
        "surfacesvfp_align_at_3",
    ): "front_mdz_01_ss_down_surfacesvfp_align_at_3",
}


@pytest.fixture
def file_content(request):
    data = np.load(PLOTTING_EXAMPLES_PATH)
    target_arr = data[PLOTTING_EXAMPLES_MAP[request.param]]
    yield target_arr


def fig_to_arr(fig: Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    rgba_buffer = fig.canvas.buffer_rgba()
    image_array = np.asarray(rgba_buffer, dtype=np.uint8)
    image_array = image_array.reshape(int(height), int(width), 4)
    return image_array


standard_plot_examples = [
    (
        ("front", "allplots_layermats_mslice_tsld"),
        {"spin_state": "up"},
        "allplots_layermats_mslice_tsld",
    ),
    (
        ("front", "allplots_layermats_mslice_align_at_2"),
        {"spin_state": "up"},
        "allplots_layermats_mslice_align_at_2",
    ),
    (
        ("front", "sldvfpplots_layermats"),
        {"spin_state": "up"},
        "sldvfpplots_layermats",
    ),
    (("front", "vfp_post"), {"spin_state": "up"}, "vfp_post"),
    (("front", "surfacesvfp"), {"spin_state": "up"}, "surfacesvfp"),
    (
        ("front", "surfacesvfp_align_at_3"),
        {"spin_state": "up"},
        "surfacesvfp_align_at_3",
    ),
    (
        ("back_ssup", "allplots_layermats_mslice_tsld"),
        {"orientation": "back", "spin_state": "up"},
        "allplots_layermats_mslice_tsld",
    ),
    (
        ("back_ssup", "allplots_layermats_mslice_align_at_2"),
        {"orientation": "back", "spin_state": "up"},
        "allplots_layermats_mslice_align_at_2",
    ),
    (
        ("back_ssup", "sldvfpplots_layermats"),
        {"orientation": "back", "spin_state": "up"},
        "sldvfpplots_layermats",
    ),
    (
        ("back_ssup", "vfp_post"),
        {"orientation": "back", "spin_state": "up"},
        "vfp_post",
    ),
    (
        ("back_ssup", "surfacesvfp"),
        {"orientation": "back", "spin_state": "up"},
        "surfacesvfp",
    ),
    (
        ("back_ssup", "surfacesvfp_align_at_3"),
        {"orientation": "back", "spin_state": "up"},
        "surfacesvfp_align_at_3",
    ),
    (
        ("front_mdz_01_ss_down", "allplots_layermats_mslice_tsld"),
        {"max_delta_z": 0.1, "spin_state": "down"},
        "allplots_layermats_mslice_tsld",
    ),
    (
        ("front_mdz_01_ss_down", "allplots_layermats_mslice_align_at_2"),
        {"max_delta_z": 0.1, "spin_state": "down"},
        "allplots_layermats_mslice_align_at_2",
    ),
    (
        ("front_mdz_01_ss_down", "sldvfpplots_layermats"),
        {"max_delta_z": 0.1, "spin_state": "down"},
        "sldvfpplots_layermats",
    ),
    (
        ("front_mdz_01_ss_down", "vfp_post"),
        {"max_delta_z": 0.1, "spin_state": "down"},
        "vfp_post",
    ),
    (
        ("front_mdz_01_ss_down", "surfacesvfp"),
        {"max_delta_z": 0.1, "spin_state": "down"},
        "surfacesvfp",
    ),
    (
        ("front_mdz_01_ss_down", "surfacesvfp_align_at_3"),
        {"max_delta_z": 0.1, "spin_state": "down"},
        "surfacesvfp_align_at_3",
    ),
]


@pytest.mark.parametrize(
    "file_content, vfp_kwargs, plot_kwarg_keys",
    standard_plot_examples,
    indirect=["file_content"],
)
def test_plot(
    vfp_inputs, plot_kwargs, file_content, vfp_kwargs, plot_kwarg_keys
):
    nslds, mslds, islds, list_of_thicknesses_gmo, list_of_roughnesses_gmo = (
        vfp_inputs
    )
    vfp = refnxVFP(
        nslds=nslds,
        thicknesses=list_of_thicknesses_gmo,
        roughnesses=list_of_roughnesses_gmo,
        mslds=mslds,
        islds=islds,
        **vfp_kwargs,
    )
    fig, _ = vfp.plot(**plot_kwargs[plot_kwarg_keys])
    fig_arr = fig_to_arr(fig)
    expected_result = file_content
    np.testing.assert_allclose(fig_arr, expected_result)


@pytest.mark.parametrize(
    "file_content, vfp_kwargs, plot_kwarg_keys",
    standard_plot_examples[:1],
    indirect=["file_content"],
)
def test_plot_supply_fig(
    vfp_inputs, plot_kwargs, file_content, vfp_kwargs, plot_kwarg_keys
):
    nslds, mslds, islds, list_of_thicknesses_gmo, list_of_roughnesses_gmo = (
        vfp_inputs
    )
    vfp = refnxVFP(
        nslds=nslds,
        thicknesses=list_of_thicknesses_gmo,
        roughnesses=list_of_roughnesses_gmo,
        mslds=mslds,
        islds=islds,
        **vfp_kwargs,
    )
    fig = plt.figure(figsize=(8, 3 * 3))
    # reset the rng.
    plt_kwargs = plot_kwargs[plot_kwarg_keys]
    plt_kwargs["surface_plot_kwargs"]["surface_rng"] = np.random.default_rng(
        seed=42
    )
    fig, _ = vfp.plot(fig=fig, **plt_kwargs)
    fig_arr = fig_to_arr(fig)
    expected_result = file_content
    np.testing.assert_allclose(fig_arr, expected_result)


@pytest.mark.parametrize(
    "file_content, vfp_kwargs, plot_kwarg_keys",
    standard_plot_examples[:1],
    indirect=["file_content"],
)
def test_plot_fail_posteriors(
    vfp_inputs, plot_kwargs, file_content, vfp_kwargs, plot_kwarg_keys
):
    plt.close("all")
    nslds, mslds, islds, list_of_thicknesses_gmo, list_of_roughnesses_gmo = (
        vfp_inputs
    )
    vfp = refnxVFP(
        nslds=nslds,
        thicknesses=list_of_thicknesses_gmo,
        roughnesses=list_of_roughnesses_gmo,
        mslds=mslds,
        islds=islds,
        **vfp_kwargs,
    )
    fig = plt.figure(figsize=(8, 3 * 3))
    orig_p_samps = plot_kwargs[plot_kwarg_keys]["posterior_samples"]
    orig_p_samps_vals = list(orig_p_samps.values())
    orig_p_samps_vals[0] = np.append(orig_p_samps_vals, [1])
    plot_kwargs[plot_kwarg_keys]["posterior_samples"] = {
        ky: val
        for ky, val in zip(
            orig_p_samps.keys(), orig_p_samps_vals, strict=False
        )
    }
    with pytest.raises(ValueError) as e:
        outcome = vfp.plot(fig=fig, **plot_kwargs[plot_kwarg_keys])
        assert outcome == e


standard_vfp = VFP(nslds=(5, 4, 3), thicknesses=(0, 20), roughnesses=(3, 4))

_, ax = plt.subplots()


@pytest.mark.parametrize(
    "ax, vfp, expected_result",
    [
        pytest.param(
            ax,
            standard_vfp,
            pytest.raises(ValueError),
            id="First _plot_surfaces test.",
        ),
    ],
)
def test_plot_surfaces(ax, vfp, expected_result):
    with expected_result as e:
        output = PlotType("surfaces")._plot_surfaces(
            ax, vfp, align_at_interface=0, surface_points=0
        )
        assert_allclose(output, e)


@pytest.mark.parametrize(
    "value, plot_type, expected_result",
    [
        pytest.param(
            0,
            PlotType("sld"),
            nullcontext(PlotType("sld")),
            id="First AxesIndex plot type test.",
        ),
        pytest.param(
            0,
            "sld",
            pytest.raises(TypeError),
            id="second AxesIndex plot type test.",
        ),
    ],
)
def test_set_axesindex(value, plot_type, expected_result):
    with expected_result as e:
        ai = AxesIndex(value)
        ai.plot_type = plot_type
        output = ai.plot_type
        assert output == e


rng = np.random.default_rng(seed=1)
first_interface = stats.norm.rvs(
    loc=0,
    scale=3,
    size=50,
    random_state=rng,
)
expected_outcome_second_test = np.vstack(
    (first_interface, first_interface + 20)
)


@pytest.mark.parametrize(
    "vfp, align_at_interface, expected_result",
    [
        pytest.param(
            standard_vfp,
            10,
            pytest.raises(ValueError),
            id="First surfaces_for_display test.",
        ),
        pytest.param(
            VFP(
                nslds=(5, 4, 3),
                thicknesses=(0, 20),
                roughnesses=(3, "conformal"),
            ),
            0,
            nullcontext(expected_outcome_second_test),
            id="Second surfaces_for_display test.",
        ),
    ],
)
def test_surfaces_for_display(vfp, align_at_interface, expected_result):
    with expected_result as e:
        res = surfaces_for_display(
            vfp,
            50,
            np.random.default_rng(1),
            align_at_interface,  # dont use same rng, reset.
        )
        assert_allclose(res, e)
