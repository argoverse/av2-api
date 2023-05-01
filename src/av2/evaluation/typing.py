"""Types for evaluation."""

from typing import Any, Dict, List

Frame = Dict[str, Any]
Frames = List[Frame]
Sequences = Dict[str, Frames]
ForecastSequences = Dict[str, Dict[int, List[Frame]]]
