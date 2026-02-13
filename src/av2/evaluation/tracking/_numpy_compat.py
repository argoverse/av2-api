"""Compatibility shim to restore deprecated NumPy scalar aliases.

Some third-party packages (e.g., TrackEval) still reference aliases like
`np.float` that were removed in recent NumPy releases. Importing this module
early (before those libraries are imported) restores safe fallbacks.
"""

from __future__ import annotations

import numpy as _np

# Map of deprecated alias -> safe replacement
_ALIASES = {
    "float": float,
    "int": int,
    "bool": bool,
    "complex": complex,
    "object": object,
    "str": str,
}

for _name, _typ in _ALIASES.items():
    if not hasattr(_np, _name):
        setattr(_np, _name, _typ)

# Also provide np.long (rare) and np.unicode if missing
if not hasattr(_np, "long"):
    setattr(_np, "long", int)
if not hasattr(_np, "unicode"):
    setattr(_np, "unicode", str)
