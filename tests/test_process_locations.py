# test_process_locations.py
import sys
import os
import pandas as pd
import pytest

# Ensure that the project root directory is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from process_locations import process_csv

def test_no_file():
    """
    If the CSV file does not exist, process_csv should raise FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        process_csv("non_existing_file.csv")

def test_minimal_csv(tmp_path):
    """
    Test with a small CSV that has the necessary columns, but minimal data.
    Check that the function runs and returns something.
    """
    # Create minimal CSV with date format MM/DD/YY HH:MM
    csv_content = """Page Number,Item Number,Local Date & Time,Latitude,Longitude,Time Zone,County,State,Country,Record Type
1,1,01/01/21 10:00,40.0,-73.0,America/New_York,Suffolk,New York,USA,Voice
2,2,01/01/21 10:05,40.0,-73.0,America/New_York,Suffolk,New York,USA,Data
"""
    csv_file = tmp_path / "test_minimal.csv"
    csv_file.write_text(csv_content, encoding='utf-8')

    # Call process_csv
    intervals = process_csv(
        csv_path=str(csv_file),
        weight_voice=2,
        weight_sms=1,
        weight_data=1,
        time_gap_minutes=15
    )

    # Debugging: Print intervals to understand what's being returned
    print("Test Minimal CSV - Intervals:", intervals)

    # We expect one interval because the time difference is 5 minutes (<= 15)
    assert len(intervals) == 1, "Should group both rows into one interval"

    first_interval = intervals[0]
    assert first_interval['state'] == 'New York'
    assert first_interval['confidence'] == 100.0
    assert first_interval['count_rows'] == 2

def test_duplicates(tmp_path):
    """
    Verify that duplicate rows are removed.
    We'll create a CSV with two identical rows and see if the function deduplicates them.
    """
    csv_content = """Page Number,Item Number,Local Date & Time,Latitude,Longitude,Time Zone,County,State,Country,Record Type
1,1,01/01/21 09:00,40.0,-73.0,America/New_York,Suffolk,New York,USA,Data
1,2,01/01/21 09:00,40.0,-73.0,America/New_York,Suffolk,New York,USA,Data
"""  # same lat/lon, same time, same state, same Record Type

    csv_file = tmp_path / "test_duplicates.csv"
    csv_file.write_text(csv_content, encoding='utf-8')

    intervals = process_csv(str(csv_file), time_gap_minutes=15)

    # Debugging: Print intervals to understand what's being returned
    print("Test Duplicates - Intervals:", intervals)

    # Because both rows are duplicates, after deduplication there should be 1 row in the group
    assert len(intervals) == 1
    first_interval = intervals[0]
    assert first_interval['count_rows'] == 1, "Duplicates should be removed, leaving only 1 record in that interval"

def test_interpolation(tmp_path):
    """
    Test interpolation: if we have an unknown in the middle row, 
    but the previous and next have the same state, and time difference is within the gap,
    it should interpolate that state.
    """
    csv_content = """Page Number,Item Number,Local Date & Time,Latitude,Longitude,Time Zone,County,State,Country,Record Type
1,1,01/01/21 08:00,40.0,-73.0,America/New_York,Westchester,New York,USA,Data
1,2,01/01/21 08:07,40.0,-73.0,America/New_York,unknown,unknown,USA,Data
1,3,01/01/21 08:15,40.0,-73.0,America/New_York,Westchester,New York,USA,Data
"""
    csv_file = tmp_path / "test_interpolation.csv"
    csv_file.write_text(csv_content, encoding='utf-8')

    intervals = process_csv(str(csv_file), time_gap_minutes=15)

    # Debugging: Print intervals to understand what's being returned
    print("Test Interpolation - Intervals:", intervals)

    # All 3 rows are within 15 minutes => 1 interval
    assert len(intervals) == 1, "Should group all three rows into one interval"
    intvl = intervals[0]
    assert intvl['count_rows'] == 3
    # The middle row should have been interpolated => final state is "New York"
    assert intvl['state'] == 'New York'
    # Confidence should be approximately 100%
    assert intvl['confidence'] == pytest.approx(100.0, 0.1)

def test_missing_columns(tmp_path):
    """
    If a required column is missing, it should raise ValueError.
    """
    # Remove the 'State' column
    csv_content = """Page Number,Item Number,Local Date & Time,Latitude,Longitude,Time Zone,County,Country,Record Type
1,1,01/01/21 10:00,40.0,-73.0,America/New_York,Suffolk,USA,Voice
"""
    csv_file = tmp_path / "test_missing.csv"
    csv_file.write_text(csv_content, encoding='utf-8')

    with pytest.raises(ValueError, match="Missing required column"):
        process_csv(str(csv_file))
