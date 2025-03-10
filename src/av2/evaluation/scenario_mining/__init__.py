# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Dataset evaluation subpackage."""

from enum import Enum, unique
from typing import Final


@unique
class ScenarioMiningCategories(str, Enum):
    """Sensor dataset annotation categories."""

    REFERRED_OBJECT = "REFERRED_OBJECT"
    RELATED_OBJECT = "RELATED_OBJECT"
    OTHER_OBJECT = "OTHER_OBJECT"


"""Constants for scenario mining challenge."""
SUBMETRIC_TO_METRIC_CLASS_NAME: Final = {
    "MOTA": "CLEAR",
    "HOTA": "HOTA",
}

AV2_CATEGORIES: Final = tuple(x.value for x in ScenarioMiningCategories)
