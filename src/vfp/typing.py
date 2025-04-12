"""
Define a type alias `ParameterLike` which is conditional on the available
dependencies to `vfp`. `ParameterLike` is the union of `float`, `int`,
`refnx.analysis.Parameter`, `refnx.analysis.parameter._BinaryOp`,
`bumps.parameter.Parameter` & `bumps.parameter.Expression`
"""

# not clean but the following appears to work with refl1d + refnx present.
type ParameterLike = int | float

try: # if we have bumps, see if we can load in refnx too.
    from bumps.parameter import Parameter as bumpsParameter, Expression
    type ParameterLike = int | float | bumpsParameter | Expression
    try:
        from refnx.analysis import Parameter as refnxParameter
        from refnx.analysis.parameter import _BinaryOp
        type ParameterLike = int | float | bumpsParameter | Expression | refnxParameter | _BinaryOp
    except ImportError as ie:
        print(f"{ie} compatible refnx package not installed.")
except ImportError as ie: # if we don't have bumps, try refnx.
    print(f"{ie} compatible refl1d & bumps packages not installed.")
    try:
        from refnx.analysis import Parameter as refnxParameter
        from refnx.analysis.parameter import _BinaryOp
        type ParameterLike = int | float | refnxParameter | _BinaryOp
    except ImportError as ie:
        print(f"{ie} compatible refnx package not installed.")
        