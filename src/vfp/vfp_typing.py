"""Types for use with vfp package."""

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NotRequired,
    Protocol,
    TypedDict,
)

import numpy as np

"""
Define a type alias ``ParameterLike`` which is conditional on the available
dependencies to ``vfp``. ``ParameterLike`` is the union of ``float``, ``int``,
``refnx.analysis.Parameter``, ``refnx.analysis.parameter._BinaryOp``,
``bumps.parameter.Parameter`` & ``bumps.parameter.Expression``
"""
if TYPE_CHECKING:
    from bumps.parameter import Expression as bumpsExpr
    from bumps.parameter import Parameter as bumpsParam
    from refnx.analysis import Parameter as refnxParam
    from refnx.analysis.parameter import _BinaryOp as refnxOp
else:
    # at run time define as Any and validate within classes.
    bumpsexpr = bumpsparam = refnxparam = refnxop = Any

type ParameterLike = (
    int | float | bumpsParam | bumpsExpr | refnxParam | refnxOp
)

type VFPAttrType = dict[
    str,
    np.typing.NDArray[np.float64]
    | str
    | float
    | None
    | SldConstraintType
    | Literal["none", "up", "down", "front", "back"],
]


# define a type for user defined SldConstraints to follow.
class SldConstraintType(Protocol):
    """SldConstraint is a user-defined class that implements these methods."""

    def __init__(self, required_pars: dict[str, ParameterLike]) -> None:
        """Instantiate user defined SldConstraint class.

        Parameters
        ----------
        required_pars : dict[str, ParameterLike]
            Parameters that are required in the user-defined constraint.
        """
        ...

    def layer_choices(self) -> list[int] | tuple[int, ...]:
        """Select which layer slds should be constrained."""
        ...

    def __call__(
        self, layer_integrals: list[float] | tuple[float, ...]
    ) -> tuple[
        list[int] | tuple[int, ...],
        list[ParameterLike] | tuple[ParameterLike, ...],
    ]:
        """Implement a method to constrain the slds.

        Parameters
        ----------
        layer_integrals : list[float] | tuple[float, ...]
            Integrals of layer volume fraction profiles used in user
            constraint.
        """
        ...


@dataclass
class LayerMaterialFraction:
    """The volume fraction of a particular material in a particular layer."""

    name: str
    """Name of material within layer."""
    volume_fraction: ParameterLike
    """Volume fraction of material in the layer."""


class SldPlotKwargType(TypedDict):
    """Kwargs to be passed to ``plotting.PlotType._plot_sld``."""

    microslice: NotRequired[bool]
    """Flag to plot sld as microsliced slabs as modelled in fitting engine.
    If False, continuous sld is plotted as calculated from vfp. By default
    True."""
    total_sld: NotRequired[bool]
    """If true, plots sldn +/- sldm. Else, plots sldn, sldm separately. By
    default, False."""


class VfpPlotKwargType(TypedDict, total=False):
    """Kwargs to be passed to ``plotting.PlotType._plot_vfp``."""

    layer_materials: NotRequired[dict[int, LayerMaterialFraction]]
    """Each key is the layer number (e.g fronting = 0), while the value should
    be a ``LayerMaterialFraction`` dict, where the keys are the material
    names, and values are ``ParameterLike`` (float, int, refnxParameter,
    BumpsParameter). The material names are used as labels, and will overwrite
    the `labels` kwarg."""
    colours: NotRequired[tuple[tuple[float, float, float], ...]]
    """Colours to plot vfp profile. Posterior samples are plotted in every
    second colour, while the nominal profile of each layer is plotted in every
    odd colour."""
    total_vf: NotRequired[bool]
    """If true, plots the total_vf of the representative profiles by summing
    across all layers' volume fractions."""
    labels: NotRequired[list[str]]
    """Labels to be applied to the legend of the volume fraction profile.
    Order of labels should match the order of layers in vfp, from
    fronting to backing.."""


class SurfacePlotKwargType(TypedDict):
    """Kwargs to be passed to ``plotting.PlotType._plot_surfaces``."""

    surface_points: NotRequired[int]
    """Number of points to simulate across each interface."""
    surface_rng: NotRequired[np.random.Generator]
    """Random number generator for producing draws from each interface's
    modelled distribution. If supplied, will generate deterministic draws
    so that the results are repeatable. If not supplied, a random seed will be
    set when calling this function."""
    surface_colours: NotRequired[tuple[tuple[float, float, float], ...]]
    """Colours of interfaces."""
