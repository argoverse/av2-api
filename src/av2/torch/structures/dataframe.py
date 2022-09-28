"""Backend agnostic DataFrame abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import List, Tuple, Union, cast

from pyarrow import feather
from upath import UPath

from av2.utils.typing import NDArrayBool, NDArrayNumber, PathType

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

try:
    import polars as pl

    pl.Config.with_columns_kwargs = True

    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False


@unique
class DataFrameBackendType(str, Enum):
    """DataFrame compute backends."""

    PANDAS = "PANDAS"
    POLARS = "POLARS"


@dataclass
class DataFrame:
    """Backend agnostic dataframe."""

    frame: Union[pd.DataFrame, pl.DataFrame]
    backend: DataFrameBackendType = DataFrameBackendType.PANDAS

    def __post_init__(self) -> None:
        if self.backend == DataFrameBackendType.PANDAS and not _PANDAS_AVAILABLE:
            raise RuntimeError("`pandas` must be installed to use this backend!")
        if self.backend == DataFrameBackendType.POLARS and not _POLARS_AVAILABLE:
            raise RuntimeError("`polars` must be installed to use this backend!")

    def __getitem__(self, index: Union[str, List[str], pd.DataFrame, DataFrame]) -> DataFrame:
        """Get the item at the specified indices.

        Args:
            index: Mask or index.

        Returns:
            DataFrame with the selected elements.
        """
        if self.backend == DataFrameBackendType.POLARS:
            dataframe_polars: pl.DataFrame = self.frame.select(pl.col(index))
            return DataFrame(dataframe_polars, backend=self.backend)
        if isinstance(index, (List, str)):
            return DataFrame(self.frame[index], backend=self.backend)

        # TODO: Better communicate types.
        index_npy: NDArrayBool = index.to_numpy()
        return DataFrame(self.frame[index_npy], backend=self.backend)

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the DataFrame."""
        return self.frame.shape

    @property
    def columns(self) -> List[str]:
        """Return the columns of the DataFrame."""
        return list(self.frame.columns)

    @classmethod
    def read(cls, path: PathType, backend: DataFrameBackendType = DataFrameBackendType.PANDAS) -> DataFrame:
        """Read a feather file into a DataFrame.

        Args:
            path: Source feather file path.
            backend: DataFrame backend for initialization.

        Returns:
            The DataFrame.

        Raises:
            NotImplementedError: If the dataframe backend is not implemented.
        """
        if backend == DataFrameBackendType.PANDAS:
            if not isinstance(path, UPath):
                dataframe_pandas = cast(pd.DataFrame, feather.read_feather(str(path), memory_map=True))
                return cls(dataframe_pandas, backend=backend)
            with path.open("rb") as file_handle:
                dataframe_pandas = cast(pd.DataFrame, feather.read_feather(file_handle))
            return cls(dataframe_pandas, backend=backend)
        if backend == DataFrameBackendType.POLARS:
            dataframe_polars = pl.read_ipc(file_handle, memory_map=True)
            return cls(dataframe_polars, backend=backend)
        raise NotImplementedError("This backend is not implemented!")

    def write(self, path: PathType) -> None:
        """Write the DataFrame to a feather file.

        Args:
            path: Feather file destination path.
        """
        with path.open("wb") as file_handle:
            if self.backend == DataFrameBackendType.PANDAS:
                feather.write_feather(self.frame, file_handle, compression=None)
            elif self.backend == DataFrameBackendType.POLARS:
                self.frame.write_ipc(file_handle)
            else:
                raise NotImplementedError("This backend is not implemented!")

    @classmethod
    def concat(cls, dataframe_list: List["DataFrame"], axis: int = 0) -> DataFrame:
        """Concatenate a list of DataFrames into one DataFrame.

        Args:
            dataframe_list: List of dataframes.
            axis: Axis of concatenation.

        Returns:
            The DataFrame.

        Raises:
            NotImplementedError: If the dataframe backend is not implemented.
        """
        dataframe_backend_type = dataframe_list[0].backend
        if dataframe_backend_type == DataFrameBackendType.PANDAS:
            assert all(isinstance(dataframe.frame, pd.DataFrame) for dataframe in dataframe_list)
            _dataframe_pandas_list: List[pd.DataFrame] = [
                cast(pd.DataFrame, dataframe.frame) for dataframe in dataframe_list
            ]
            dataframe_pandas: pd.DataFrame = pd.concat(_dataframe_pandas_list, axis=axis).reset_index(drop=True)
            return cls(dataframe_pandas, backend=dataframe_backend_type)
        if dataframe_backend_type == DataFrameBackendType.POLARS:
            assert all(isinstance(dataframe.frame, pl.DataFrame) for dataframe in dataframe_list)
            _dataframe_polars_list: List[pl.DataFrame] = [
                cast(pl.DataFrame, dataframe.frame) for dataframe in dataframe_list
            ]
            dataframe_polars = pl.concat(_dataframe_polars_list, how="vertical" if axis == 0 else "horizontal")
            return cls(dataframe_polars, backend=dataframe_backend_type)
        raise NotImplementedError("This backend is not implemented!")

    @classmethod
    def from_numpy(
        cls, arr_npy: NDArrayNumber, columns: List[str], backend: DataFrameBackendType = DataFrameBackendType.PANDAS
    ) -> DataFrame:
        """Convert the numpy ndarray into a DataFrame.

        Args:
            arr_npy: Numpy array.
            columns: List of column names.
            backend: DataFrame backend for initialization.

        Returns:
            The DataFrame.

        Raises:
            NotImplementedError: If the dataframe backend is not implemented.
        """
        if backend == DataFrameBackendType.PANDAS:
            dataframe_pandas = pd.DataFrame(arr_npy, columns=columns)
            return cls(dataframe_pandas)
        if backend == DataFrameBackendType.POLARS:
            dataframe_polars = pl.DataFrame(arr_npy, columns=columns)
            return cls(dataframe_polars)
        raise NotImplementedError("This backend is not implemented!")

    def to_numpy(self) -> NDArrayNumber:
        """Convert the DataFrame into a numpy ndarray."""
        arr_npy: NDArrayNumber = self.frame.to_numpy()
        return arr_npy

    def __ge__(self, other: object) -> DataFrame:
        if not isinstance(other, (DataFrame, float, int)):
            return NotImplemented
        return DataFrame(self.frame >= other, backend=self.backend)

    def __gt__(self, other: object) -> DataFrame:
        if not isinstance(other, (DataFrame, float, int)):
            return NotImplemented
        return DataFrame(self.frame > other, backend=self.backend)

    def __le__(self, other: object) -> DataFrame:
        if not isinstance(other, (DataFrame, float, int)):
            return NotImplemented
        return DataFrame(self.frame <= other, backend=self.backend)

    def __lt__(self, other: object) -> DataFrame:
        if not isinstance(other, (DataFrame, float, int)):
            return NotImplemented
        return DataFrame(self.frame < other, backend=self.backend)

    def __eq__(self, other: object) -> DataFrame:
        if not isinstance(other, (DataFrame, float, int)):
            return NotImplemented
        return DataFrame(self.frame == other, backend=self.backend)

    def __and__(self, other: object) -> DataFrame:
        if not isinstance(other, DataFrame):
            return NotImplemented
        return DataFrame(self.frame & other.frame, backend=self.backend)

    def sort(self, columns: List[str]) -> DataFrame:
        """Sort the DataFrame with respect to the columns.

        Args:
            columns: List of column names.

        Returns:
            The sorted DataFrame.

        Raises:
            NotImplementedError: If the dataframe backend is not implemented.
        """
        if self.backend == DataFrameBackendType.PANDAS:
            dataframe_pandas: pd.DataFrame = self.frame.sort_values(columns)
            return DataFrame(dataframe_pandas)
        if self.backend == DataFrameBackendType.POLARS:
            dataframe_polars: pl.DataFrame = self.frame.sort(columns)
            return DataFrame(dataframe_polars)
        raise NotImplementedError("This backend is not implemented!")
