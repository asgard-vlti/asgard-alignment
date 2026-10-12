import logging
import os
import time
import unittest
from datetime import datetime, timezone
from unittest import mock

from asgard_alignment.logging_utils import utc_log_formatter


class UTCLoggingTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(time, "tzset"), "Requires POSIX timezone support")
    def test_utc_records_are_independent_of_host_timezone(self):
        formatter = utc_log_formatter()
        try:
            for zone in ("UTC", "America/Santiago"):
                with self.subTest(zone=zone), mock.patch.dict(os.environ, {"TZ": zone}):
                    time.tzset()
                    for month in (1, 7):
                        instant = datetime(2026, month, 1, 0, 0, 0, tzinfo=timezone.utc)
                        record = logging.makeLogRecord({
                            "created": instant.timestamp(),
                            "msecs": 123,
                            "levelname": "INFO",
                            "msg": "test record",
                            "args": (),
                        })
                        self.assertEqual(
                            formatter.format(record),
                            f"2026-{month:02d}-01T00:00:00.123Z INFO test record",
                        )
        finally:
            time.tzset()


if __name__ == "__main__":
    unittest.main()
