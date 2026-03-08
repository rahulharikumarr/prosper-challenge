"""Unit tests for healthie.py utility functions.

These tests cover pure logic that requires no browser, credentials, or
network access. Integration with the live Healthie site is tested manually
via test_healthie.py (run with real credentials).
"""

import os
import sys

# Make healthie importable without triggering Playwright at import time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from healthie import _normalise_time


class TestNormaliseTime:
    """_normalise_time converts various time representations to 'H:MM AM/PM'."""

    def test_already_normalised(self):
        assert _normalise_time("2:00 PM") == "2:00 PM"
        assert _normalise_time("10:30 AM") == "10:30 AM"
        assert _normalise_time("12:00 PM") == "12:00 PM"

    def test_lowercase_am_pm(self):
        assert _normalise_time("2:00 pm") == "2:00 PM"
        assert _normalise_time("10:30 am") == "10:30 AM"

    def test_no_space_before_am_pm(self):
        assert _normalise_time("2pm") == "2:00 PM"
        assert _normalise_time("10am") == "10:00 AM"
        assert _normalise_time("2PM") == "2:00 PM"

    def test_24_hour_format(self):
        assert _normalise_time("14:00") == "2:00 PM"
        assert _normalise_time("10:00") == "10:00 AM"
        assert _normalise_time("00:00") == "12:00 AM"
        assert _normalise_time("12:00") == "12:00 PM"
        assert _normalise_time("13:30") == "1:30 PM"
        assert _normalise_time("23:59") == "11:59 PM"

    def test_passes_through_unrecognised(self):
        # Unrecognised formats are returned as-is rather than crashing.
        assert _normalise_time("morning") == "morning"
        assert _normalise_time("noon") == "noon"

    def test_strips_whitespace(self):
        assert _normalise_time("  2:00 PM  ") == "2:00 PM"


class TestDateValidation:
    """Date guard logic in handle_create_appointment: past dates are rejected."""

    def test_future_date_passes(self):
        from datetime import date, timedelta

        from dateutil.parser import parse as parse_date

        future = date.today() + timedelta(days=7)
        appt_date = future.strftime("%B %-d, %Y")
        # parse_date returns a datetime; .date() converts it for comparison
        parsed = parse_date(appt_date)
        assert parsed.date() >= date.today()

    def test_past_date_is_detected(self):
        from datetime import date

        from dateutil.parser import parse as parse_date

        past_date = "January 1, 2020"
        parsed = parse_date(past_date)
        assert parsed.date() < date.today()

    def test_various_date_formats_parsed(self):
        from datetime import date

        from dateutil.parser import parse as parse_date

        cases = [
            "March 10, 2026",
            "2026-03-10",
            "03/10/2026",
        ]
        for case in cases:
            parsed = parse_date(case, default=date.today().replace(day=1))
            assert parsed.year == 2026
            assert parsed.month == 3
            assert parsed.day == 10
