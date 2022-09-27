from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import List, Union

import pandas as pd
import polars as pl
from pyarrow import feather

from av2.utils.typing import DataFrameType, NDArrayNumber, PathType


@unique
class DataFrameBackendType(str, Enum):

    PANDAS = "PANDAS"
    POLARS = "POLARS"


@dataclass
class DataFrame:
    """Backend agnostic dataframe."""

    _dataframe: Union[pd.DataFrame, pl.DataFrame]
    _backend: DataFrameBackendType = DataFrameBackendType.PANDAS

    def __getitem__(self, index):
        if self._backend == DataFrameBackendType.POLARS:
            return DataFrame(self._dataframe.select(pl.col(index)), _backend=self._backend)
        if isinstance(index, (List, str)):
            return DataFrame(self._dataframe[index], _backend=self._backend)
        return DataFrame(self._dataframe[index.to_numpy()], _backend=self._backend)

    @property
    def shape(self):
        return self._dataframe.shape

    @property
    def columns(self):
        return self._dataframe.columns

    @classmethod
    def read(cls, path: PathType, backend: DataFrameBackendType = DataFrameBackendType.PANDAS) -> DataFrame:
        with path.open("rb") as file_handle:
            if backend == DataFrameBackendType.PANDAS:
                dataframe_pandas: pd.DataFrame = feather.read_feather(file_handle, memory_map=True)
                return cls(dataframe_pandas, _backend=backend)
            if backend == DataFrameBackendType.POLARS:
                dataframe_polars = pl.read_ipc(file_handle, memory_map=True)
                return cls(dataframe_polars, _backend=backend)
            raise NotImplementedError("This backend is not implemented!")

    def write(self, path: PathType) -> None:
        with path.open("wb") as file_handle:
            if self._backend == DataFrameBackendType.PANDAS:
                feather.write_feather(self._dataframe, file_handle, compression=None)
            if self._backend == DataFrameBackendType.POLARS:
                self._dataframe.write_ipc(file_handle)
            raise NotImplementedError("This backend is not implemented!")

    # def as_pandas(self) -> pd.DataFrame:
    #     if self.backend == DataFrameBackendType.PANDAS:
    #         return self._dataframe
    #     elif self.backend == DataFrameBackendType.POLARS:
    #         dataframe = self._dataframe.to_pandas()
    #         return dataframe

    @classmethod
    def concat(cls, dataframe_list: List["DataFrame"], axis: int = 0) -> DataFrame:
        dataframe_backend_type = dataframe_list[0]._backend
        if dataframe_backend_type == DataFrameBackendType.PANDAS:
            _dataframe_pandas_list: List[pd.DataFrame] = [dataframe._dataframe for dataframe in dataframe_list]
            dataframe_pandas: pd.DataFrame = pd.concat(_dataframe_pandas_list, axis=axis).reset_index(drop=True)
            return cls(dataframe_pandas, _backend=dataframe_backend_type)
        if dataframe_backend_type == DataFrameBackendType.POLARS:
            _dataframe_polars_list: List[pl.DataFrame] = [dataframe._dataframe for dataframe in dataframe_list]
            dataframe_polars = pl.concat(_dataframe_polars_list, how="vertical" if axis == 0 else "horizontal")
            return cls(dataframe_polars, _backend=dataframe_backend_type)

    @classmethod
    def from_numpy(
        cls, arr: NDArrayNumber, columns: List[str], backend: DataFrameBackendType = DataFrameBackendType.PANDAS
    ) -> DataFrame:
        if backend == DataFrameBackendType.PANDAS:
            dataframe_pandas = pd.DataFrame(arr, columns=columns)
            return cls(dataframe_pandas)
        if backend == DataFrameBackendType.POLARS:
            dataframe_polars = pl.DataFrame(arr, columns=columns)
            return cls(dataframe_polars)
        raise NotImplementedError("This backend is not implemented!")

    def to_numpy(self):
        return self._dataframe.to_numpy()

    def __gt__(self, x):
        return DataFrame(self._dataframe > x, _backend=self._backend)

    def __le__(self, x):
        return DataFrame(self._dataframe < x, _backend=self._backend)

    def __eq__(self, x):
        return DataFrame(self._dataframe == x, _backend=self._backend)

    def __and__(self, x):
        return DataFrame(self._dataframe & x._dataframe, _backend=self._backend)

    def sort(self, columns: List[str]) -> DataFrame:
        if self._backend == DataFrameBackendType.PANDAS:
            dataframe_pandas: pd.DataFrame = self._dataframe.sort_values(columns)
            return DataFrame(dataframe_pandas)
        if self._backend == DataFrameBackendType.POLARS:
            dataframe_polars: pl.DataFrame = self._dataframe.sort(columns)
            return DataFrame(dataframe_polars)
        raise NotImplementedError("This backend is not implemented!")


# def dataframe_read_feather(path: PathType, use_pyarrow: bool = True) -> DataFrameType:
#     """Read a feather file and load it as a `polars` dataframe.

#     Args:
#         path: Path to the feather file.

#     Returns:
#         The feather file as a `polars` dataframe.
#     """
#     with path.open("rb") as f:
#         if use_pyarrow:
#             return feather.read_feather(f, memory_map=True)
#         return pl.read_ipc(f, use_pyarrow=False, memory_map=True)


# def dataframe_write_feather(path: PathType, dataframe: DataFrameType, use_pyarrow: bool = True) -> None:
#     with path.open("rb") as f:
#         if use_pyarrow:
#             feather.write_feather(dataframe, f, compression="uncompressed")
#         else:
#             dataframe.write_ipc(f)


def dataframe_concat(dataframes: List[DataFrameType], axis: int = 0) -> DataFrameType:
    if all(isinstance(dataframe, pd.DataFrame) for dataframe in dataframes):
        dataframes_pandas = cast(List[pd.DataFrame], dataframes)
        dataframe_pandas: pd.DataFrame = pd.concat(dataframes_pandas, axis=axis).reset_index(drop=True)
        return dataframe_pandas
    if all(isinstance(dataframe, pl.DataFrame) for dataframe in dataframes):
        dataframes_polars = cast(List[pl.DataFrame], dataframes)
        dataframe_polars: pl.DataFrame = pl.concat(dataframes_polars, how="vertical" if axis == 0 else "horizontal")
        return dataframe_polars
    raise RuntimeError("Dataframes must all be the same type.")


def dataframe_from_numpy(arr: NDArrayNumber, columns: List[str], use_pandas: bool = True):
    if use_pandas:
        return pd.DataFrame(arr, columns=columns)
    return pl.DataFrame(arr, columns=columns)


def dataframe_sort(dataframe: DataFrameType, columns: List[str]):
    if isinstance(dataframe, pd.DataFrame):
        return dataframe.sort_values(columns)
    return dataframe.sort(columns)
