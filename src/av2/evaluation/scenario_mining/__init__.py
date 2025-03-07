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
