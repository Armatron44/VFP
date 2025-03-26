"""
Setup to define a type alias `ParameterLike` which is conditional
on the available dependencies to `vfp`.
"""

# not clean but the following appears to work with refl1d + refnx present.
type ParameterLike = int | float

try: # if we have bumps, see if we can load in refnx too.
    from bumps.parameter import Parameter as bumpsParameter
    type ParameterLike = int | float | bumpsParameter
    try:
        from refnx.analysis import Parameter as refnxParameter
        type ParameterLike = int | float | bumpsParameter | refnxParameter
    except ImportError as ie:
        print(f"{ie} compatible refnx package not installed.")
except ImportError as ie: # if we don't have bumps, try refnx.
    print(f"{ie} compatible refl1d & bumps packages not installed.")
    try:
        from refnx.analysis import Parameter as refnxParameter
        type ParameterLike = int | float | refnxParameter
    except ImportError as ie:
        print(f"{ie} compatible refnx package not installed.")
        