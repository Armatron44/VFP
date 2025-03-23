# read version from installed package
from importlib.metadata import version

__version__ = version("vfp")

from vfp.vfp import VFP

try:
    from vfp.vfp import refnxVFP

    __all__ = ["VFP", "refnxVFP"]
    try:
        from vfp.vfp import refl1dVFP
        __all__ = ["VFP", "refnxVFP", "refl1dVFP"]
    except ImportError as ie:
        print(f"{ie} compatible refl1d package not installed.")
except ImportError as ie:
    print(f"{ie} compatible refnx package not installed.")
else:
    __all__ = ["VFP"]
