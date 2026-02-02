"""Parsing package for CSV ingestion and date handling."""

from .year_inference import infer_year
from .csv_parser import parse_csv, validate_dataframe
