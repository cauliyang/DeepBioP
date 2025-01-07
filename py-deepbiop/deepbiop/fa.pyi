# This file is automatically generated by pyo3_stub_gen
# ruff: noqa: E501, F401

import os
import pathlib
import typing

class FaEncoderOption:
    qual_offset: int
    bases: list[int]
    threads: int
    def __new__(cls, qual_offset: int, bases: str, threads: int | None): ...

class ParquetEncoder:
    def __new__(cls, option: FaEncoderOption): ...

class RecordData:
    id: str
    seq: str
    def __new__(cls, id: str, seq: str): ...
    def set_id(self, id: str) -> None: ...
    def set_seq(self, seq: str) -> None: ...

def convert_multiple_fas_to_one_fa(
    paths: typing.Sequence[str | os.PathLike | pathlib.Path],
    result_path: str | os.PathLike | pathlib.Path,
    parallel: bool,
) -> None: ...
def encode_fa_path_to_parquet(
    fa_path: str | os.PathLike | pathlib.Path,
    bases: str,
    qual_offset: int,
    result_path: str | os.PathLike | pathlib.Path | None,
) -> None: ...
def encode_fa_path_to_parquet_chunk(
    fa_path: str | os.PathLike | pathlib.Path,
    chunk_size: int,
    parallel: bool,
    bases: str,
    qual_offset: int,
) -> None: ...
def encode_fa_paths_to_parquet(
    fa_path: typing.Sequence[str | os.PathLike | pathlib.Path],
    bases: str,
    qual_offset: int,
) -> None: ...
def write_fa(
    records_data: typing.Sequence[RecordData],
    file_path: str | os.PathLike | pathlib.Path | None,
) -> None: ...
def write_fa_parallel(
    records_data: typing.Sequence[RecordData],
    file_path: str | os.PathLike | pathlib.Path,
    threads: int,
) -> None: ...
