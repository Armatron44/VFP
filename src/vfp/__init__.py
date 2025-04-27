from importlib.metadata import version

__version__ = version("vfp")  # read version from installed package

from vfp.vfp import VFP

__all__ = [VFP.__name__]

try:
    from vfp.vfp import refnxVFP

    __all__.append(refnxVFP.__name__)
except ImportError as ie:
    print(f"{ie} compatible refnx package not installed.")

try:
    from vfp.vfp import refl1dVFP

    __all__.append(refl1dVFP.__name__)
except ImportError as ie:
    print(f"{ie} compatible refl1d package not installed.")
