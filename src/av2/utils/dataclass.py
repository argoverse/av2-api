# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Utils to assist with dataclass-related operations."""

import itertools
from dataclasses import is_dataclass

import numpy as np
import pandas as pd


def dataclass_eq(base_dataclass: object, other: object) -> bool:
    """Check if base_dataclass is equal to the other object, with proper handling for numpy array fields.

    Args:
        base_dataclass: Base dataclass to compare against.
        other: Other object to compare against the base dataclass.

    Raises:
        ValueError: If base_dataclass is not an instance of a dataclass.

    Returns:
        Flag indicating whether base_dataclass and the other object are considered equal.
    """
    if not is_dataclass(base_dataclass):
        raise ValueError(f"'{base_dataclass.__class__.__name__}' is not a dataclass!")

    # Check whether the two objects point to the same instance
    if base_dataclass is other:
        return True

    # Check whether the two objects are both dataclasses of the same type
    if base_dataclass.__class__ is not other.__class__:
        return False

    # Check whether the dataclasses have equal values in all members
    base_tuple = vars(base_dataclass).values()
    other_tuple = vars(other).values()
    return all(_dataclass_member_eq(base_mem, other_mem) for base_mem, other_mem in zip(base_tuple, other_tuple))


def _dataclass_member_eq(base: object, other: object) -> bool:
    """Check if dataclass members base and other are equal, with proper handling for numpy arrays.

    Args:
        base: Base object to compare against.
        other: Other object to compare against the base object.

    Returns:
        Bool flag indicating whether objects a and b are equal.
    """
    # Objects are equal if they point to the same instance
    if base is other:
        return True

    # If both objects are lists, check equality for all members
    if isinstance(base, list) and isinstance(other, list):
        return all(_dataclass_member_eq(base_i, other_i) for base_i, other_i in itertools.zip_longest(base, other))

    # If both objects are np arrays, delegate equality check to numpy's built-in operation
    if isinstance(base, np.ndarray) and isinstance(other, np.ndarray):
        return bool(np.array_equal(base, other))

    # If both objects are pd dataframes, delegate equality check to pandas' built-in operation
    if isinstance(base, pd.DataFrame) and isinstance(other, pd.DataFrame):
        return bool(pd.DataFrame.equals(base, other))

    # Equality checks for all other types are delegated to the standard equality check
    try:
        return bool(base == other)
    except (TypeError, ValueError):
        return False
