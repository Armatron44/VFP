"""Types for use with vfp package."""

from typing import (
    Literal,
    NotRequired,
    Protocol,
    TypeAliasType,
    TypedDict,
    TypeIs,
    get_args,
)

import numpy as np

"""
Define a type alias ``ParameterLike`` which is conditional on the available
dependencies to ``vfp``. ``ParameterLike`` is the union of ``float``, ``int``,
``refnx.analysis.Parameter``, ``refnx.analysis.parameter._BinaryOp``,
``bumps.parameter.Parameter`` & ``bumps.parameter.Expression``
"""
try:
    from bumps.parameter import Expression as bumpsExpr
    from bumps.parameter import Parameter as bumpsParam

    type bumpsparameters = bumpsExpr | bumpsParam
except ImportError as ie:
    print(f"{ie} refl1d & bumps packages not installed.")
    type bumpsparameter = float | int
try:
    from refnx.analysis import Parameter as refnxParam
    from refnx.analysis.parameter import _BinaryOp as refnxOp

    type refnxparameters = refnxParam | refnxOp
except ImportError as ie:
    print(f"{ie} refnx packages not installed.")
    type refnxparameters = float | int

type ParameterLike = (float | int | bumpsparameters | refnxparameters)

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

    layer_materials: NotRequired[dict[int, dict[str, ParameterLike]]]
    """Each key is the layer number (e.g fronting = 0), while the value should
    be a dictionary. The nested dictionary should have keys that are the
    layers' material name and values that are material volume fraction. All
    keys in the nested dictionaries are used as labels, and will
    overwrite the ``labels`` kwarg in ``VfpPlotKwargType``."""
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


def flatten_composite_type_alias(tp) -> set[type]:
    """Recursively find all atomic types in a nested TypeAlias."""
    if isinstance(tp, TypeAliasType):
        return flatten_composite_type_alias(tp.__value__)
    args = get_args(tp)
    if not args:
        return {tp}
    # if a particular arg is itself a TypeAlias, we need to
    # get the atomic types within it.
    atoms = set()
    for arg in args:
        atoms.update(flatten_composite_type_alias(arg))
    return atoms


def _is_nested_tuple(t: tuple) -> TypeIs[tuple[tuple, ...]]:
    return all(isinstance(v, tuple) for v in t)


def _is_tuple(t: tuple) -> TypeIs[tuple[int | float, ...]]:
    return all(
        isinstance(v, int | float | np.integer | np.floating) for v in t
    )
