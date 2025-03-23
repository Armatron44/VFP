# read version from installed package
from importlib.metadata import version

__version__ = version("vfp")

from vfp.vfp import VFP
__all__ = ["VFP"]

try:
    from vfp.vfp import refnxVFP

    __all__ = __all__ + ["refnxVFP"]
except ImportError as ie:
    print(f"{ie} compatible refnx package not installed.")

try:
    from vfp.vfp import refl1dVFP
    __all__ = __all__ + ["refl1dVFP"]
except ImportError as ie:
    print(f"{ie} compatible refl1d package not installed.")
