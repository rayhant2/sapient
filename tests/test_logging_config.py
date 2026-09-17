import json
import logging
import unittest

from config.logging import JsonFormatter


class LoggingConfigurationTests(unittest.TestCase):
    def test_json_formatter_emits_machine_readable_fields(self):
        record = logging.LogRecord(
            name="sentient.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="Runtime ready with %d ticker",
            args=(1,),
            exc_info=None,
        )

        payload = json.loads(JsonFormatter().format(record))

        self.assertEqual(payload["level"], "INFO")
        self.assertEqual(payload["logger"], "sentient.test")
        self.assertEqual(payload["message"], "Runtime ready with 1 ticker")
        self.assertIn("timestamp", payload)

    def test_json_formatter_records_exception_type_not_exception_text(self):
        try:
            raise RuntimeError("sensitive provider detail")
        except RuntimeError:
            exc_info = __import__("sys").exc_info()
        record = logging.LogRecord(
            name="sentient.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=10,
            msg="Operation failed",
            args=(),
            exc_info=exc_info,
        )

        encoded = JsonFormatter().format(record)
        payload = json.loads(encoded)

        self.assertEqual(payload["exception_type"], "RuntimeError")
        self.assertNotIn("sensitive provider detail", encoded)


if __name__ == "__main__":
    unittest.main()
