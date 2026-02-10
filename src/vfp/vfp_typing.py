from typing import Protocol, TypedDict

import numpy as np

"""
Define a type alias `ParameterLike` which is conditional on the available
dependencies to `vfp`. `ParameterLike` is the union of `float`, `int`,
`refnx.analysis.Parameter`, `refnx.analysis.parameter._BinaryOp`,
`bumps.parameter.Parameter` & `bumps.parameter.Expression`
"""

type ParameterLike = int | float

try:  # if we have bumps, see if we can load in refnx too.
    from bumps.parameter import Expression
    from bumps.parameter import Parameter as bumpsParameter

    type ParameterLike = int | float | bumpsParameter | Expression
    try:
        from refnx.analysis import Parameter as refnxParameter
        from refnx.analysis.parameter import _BinaryOp

        type ParameterLike = (
            int
            | float
            | bumpsParameter
            | Expression
            | refnxParameter
            | _BinaryOp
        )
    except ImportError as ie:
        print(f"{ie} compatible refnx package not installed.")
except ImportError as ie:  # if we don't have bumps, try refnx.
    print(f"{ie} compatible refl1d & bumps packages not installed.")
    try:
        from refnx.analysis import Parameter as refnxParameter
        from refnx.analysis.parameter import _BinaryOp

        type ParameterLike = int | float | refnxParameter | _BinaryOp
    except ImportError as ie:
        print(f"{ie} compatible refnx package not installed.")


# define a type for user defined SldConstraints to follow.
class SldConstraintType(Protocol):
    def __init__(self, required_pars: dict[str, ParameterLike]) -> None: ...
    def layer_choices(self) -> list[int] | tuple[int, ...]: ...
    def __call__(
        self, layer_integrals: list[float] | tuple[float, ...]
    ) -> tuple[
        list[int] | tuple[int, ...],
        list[ParameterLike] | tuple[ParameterLike, ...],
    ]: ...


class LayerMaterialFraction(TypedDict):
    name: str
    solvation: ParameterLike


class SldPlotKwargType(TypedDict, total=False):
    microslice: bool
    total_sld: bool


class VfpPlotKwargType(TypedDict, total=False):
    layer_materials: dict[int, LayerMaterialFraction]
    colours: tuple[tuple[float, float, float], ...]
    total_vf: bool
    labels: list[str]


class SurfacePlotKwargType(TypedDict, total=False):
    surface_points: int
    surface_rng: np.random.Generator
    colours: tuple[tuple[float, float, float], ...]
