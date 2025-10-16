import pathlib

import numpy as np
import pytest
from matplotlib.figure import Figure

from vfp import refnxVFP

EPS = np.finfo(float).eps
rng = np.random.default_rng(seed=42)

PLOTTING_EXAMPLES_PATH = (
    pathlib.Path(__file__).parent
    / "plotting_examples"
    / "plotting_examples.npz"
)

PLOTTING_EXAMPLES_MAP = {("front"): "image_array"}


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


standard_plot_examples = [(("front"), "front")]


@pytest.mark.parametrize(
    "file_content, vfp_orientation",
    standard_plot_examples,
    indirect=["file_content"],
)
def test_plot(
    vfp_inputs,
    posterior_samples_setup,
    materials_by_layer_setup,
    file_content,
    vfp_orientation,
):
    dd_gmo_nslds, list_of_thicknesses_gmo, list_of_roughnesses_gmo = (
        vfp_inputs
    )
    dd_vfp_gmo = refnxVFP(
        nslds=dd_gmo_nslds,
        thicknesses=list_of_thicknesses_gmo,
        roughnesses=list_of_roughnesses_gmo,
        orientation=vfp_orientation,
    )

    fig, _ = dd_vfp_gmo.plot(
        posterior_samples=posterior_samples_setup,
        surface_rng=rng,
        vfp_plot_kwargs={"layer_materials": materials_by_layer_setup},
    )

    fig_arr = fig_to_arr(fig)
    expected_result = file_content
    np.testing.assert_allclose(fig_arr, expected_result)
