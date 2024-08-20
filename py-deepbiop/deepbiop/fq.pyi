# This file is automatically generated by pyo3_stub_gen
# ruff: noqa: E501, F401

import os
import pathlib
import typing

class FqEncoderOption:
    kmer_size: int
    qual_offset: int
    bases: list[int]
    vectorized_target: bool
    threads: int
    def __new__(cls,kmer_size:int, qual_offset:int, bases:str, vectorized_target:bool, threads:typing.Optional[int]): ...

class JsonEncoder:
    def __new__(cls,option:FqEncoderOption): ...
    ...

class ParquetEncoder:
    def __new__(cls,option:FqEncoderOption): ...
    ...

class Predict:
    r"""
    A struct to store the prediction result
    """
    prediction: list[int]
    seq: str
    id: str
    is_truncated: bool
    qual: typing.Optional[str]
    def __new__(cls,prediction:typing.Sequence[int], seq:str, id:str, is_truncated:bool, qual:typing.Optional[str]): ...
    def __repr__(self) -> str:
        ...

    def prediction_region(self) -> list[tuple[int, int]]:
        r"""
        Get the prediction region
        """
        ...

    def smooth_prediction(self, window_size:int) -> list[tuple[int, int]]:
        r"""
        Get the smooth prediction region
        """
        ...

    def smooth_label(self, window_size:int) -> list[int]:
        r"""
        Get the smooth label
        """
        ...

    def smooth_and_select_intervals(self, smooth_window_size:int, min_interval_size:int, append_interval_number:int) -> list[tuple[int, int]]:
        r"""
        Smooth and select intervals
        """
        ...

    def seq_len(self) -> int:
        r"""
        Get the sequence length
        """
        ...

    def qual_array(self) -> list[int]:
        r"""
        Get the quality score array
        """
        ...

    def show_info(self, smooth_interval:typing.Sequence[tuple[int, int]], text_width:typing.Optional[int]) -> str:
        r"""
        Show the information of the prediction
        """
        ...

    def __getstate__(self) -> typing.Any:
        ...

    def __setstate__(self, state:typing.Any) -> None:
        ...


class RecordData:
    id: str
    seq: str
    qual: str
    def __new__(cls,id:str, seq:str, qual:str): ...
    def set_id(self, id:str) -> None:
        ...

    def set_seq(self, seq:str) -> None:
        ...

    def set_qual(self, qual:str) -> None:
        ...


def convert_multiple_fqs_to_one_fq(paths:typing.Sequence[str | os.PathLike | pathlib.Path],result_path:str | os.PathLike | pathlib.Path,parallel:bool) -> None:
    ...

def encode_fq_path_to_json(fq_path:str | os.PathLike | pathlib.Path,k:int,bases:str,qual_offset:int,vectorized_target:bool,result_path:typing.Optional[str | os.PathLike | pathlib.Path]) -> None:
    ...

def encode_fq_path_to_parquet(fq_path:str | os.PathLike | pathlib.Path,bases:str,qual_offset:int,vectorized_target:bool,result_path:typing.Optional[str | os.PathLike | pathlib.Path]) -> None:
    ...

def encode_fq_path_to_parquet_chunk(fq_path:str | os.PathLike | pathlib.Path,chunk_size:int,parallel:bool,bases:str,qual_offset:int,vectorized_target:bool) -> None:
    ...

def encode_fq_paths_to_parquet(fq_path:typing.Sequence[str | os.PathLike | pathlib.Path],bases:str,qual_offset:int,vectorized_target:bool) -> None:
    ...

def encode_qual(qual:str,qual_offset:int) -> list[int]:
    r"""
    Convert ASCII quality to Phred score for Phred+33 encoding
    """
    ...

def fastq_to_fasta(fastq_path:str | os.PathLike | pathlib.Path,fasta_path:str | os.PathLike | pathlib.Path) -> None:
    ...

def generate_kmers(base:str,k:int) -> list[str]:
    ...

def get_label_region(labels:typing.Sequence[int]) -> list[tuple[int, int]]:
    ...

def kmers_to_seq(kmers:typing.Sequence[str]) -> str:
    ...

def normalize_seq(seq:str,iupac:bool) -> str:
    r"""
    Normalize a DNA sequence by converting any non-standard nucleotides to standard ones.
    
    This function takes a DNA sequence as a `String` and a boolean flag `iupac` indicating whether to normalize using IUPAC ambiguity codes.
    It returns a normalized DNA sequence as a `String`.
    
    # Arguments
    
    * `seq` - A DNA sequence as a `String`.
    * `iupac` - A boolean flag indicating whether to normalize using IUPAC ambiguity codes.
    
    # Returns
    
    A normalized DNA sequence as a `String`.
    
    # Examples
    
    ```
    use deepbiop_fq as fq;
    let seq = "acGTN".to_string();
    let normalized_seq = fq::normalize_seq(seq, false);
    assert_eq!(normalized_seq, "ACGTN");
    ```
    """
    ...

def seq_to_kmers(seq:str,k:int,overlap:bool) -> list[str]:
    ...

def test_predicts(predicts:typing.Sequence[Predict]) -> None:
    ...

def write_fq(records_data:typing.Sequence[RecordData],file_path:typing.Optional[str | os.PathLike | pathlib.Path]) -> None:
    ...

def write_fq_parallel(records_data:typing.Sequence[RecordData],file_path:str | os.PathLike | pathlib.Path,threads:int) -> None:
    ...

