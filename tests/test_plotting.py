import pathlib

import numpy as np
import pytest
from matplotlib.figure import Figure

from vfp import refnxVFP

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
