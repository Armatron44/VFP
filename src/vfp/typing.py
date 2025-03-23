from __future__ import annotations
from typing import TYPE_CHECKING

type ParameterLike = float

if TYPE_CHECKING:
    try:
        from bumps.parameter import Parameter as bumpsParameter
        type ParameterLike = ParameterLike | bumpsParameter
    except ImportError as ie:
        print(f"{ie} compatible refl1d & bumps packages not installed.")
    
    try:
        from refnx.analysis import Parameter as refnxParameter
        type ParameterLike = ParameterLike | refnxParameter
    except ImportError as ie:
        print(f"{ie} compatible refnx package not installed.")