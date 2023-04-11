"""Constants for tracking challenge."""

from typing import Dict, Final

from av2.evaluation.common.constants import CompetitionCategories

SUBMETRIC_TO_METRIC_CLASS_NAME: Final[Dict[str, str]] = {
    "MOTA": "CLEAR",
    "HOTA": "HOTA",
}

av2_classes: Final = tuple(x.value for x in CompetitionCategories)
