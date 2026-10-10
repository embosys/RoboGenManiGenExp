import faulthandler
import io
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from run_output import capture_console


class CaptureConsoleTest(unittest.TestCase):
    def test_faulthandler_accepts_captured_stderr(self):
        enabled = faulthandler.is_enabled()
        try:
            with tempfile.TemporaryDirectory() as directory:
                with capture_console(Path(directory) / 'run.log'):
                    self.assertEqual(sys.stderr.fileno(), sys.__stderr__.fileno())
                    faulthandler.enable(all_threads=False)
                    self.assertTrue(faulthandler.is_enabled())
        finally:
            if not enabled:
                faulthandler.disable()

    def test_retained_stream_survives_logging_shutdown(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'run.log'
            with redirect_stdout(stdout), redirect_stderr(stderr):
                with capture_console(log):
                    retained = sys.stderr
                    print('stdout message')
                    print('stderr message', file=sys.stderr)
                    retained.close()
                    print('still capturing', file=sys.stderr)
                retained.write('late message\n')
                retained.flush()
                retained.close()
            self.assertEqual(log.read_text(), 'stdout message\nstderr message\nstill capturing\n')
        self.assertFalse(stdout.closed)
        self.assertFalse(stderr.closed)
        self.assertIn('late message', stderr.getvalue())

    def test_stream_without_descriptor_preserves_unsupported_operation(self):
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stderr(io.StringIO()):
                with capture_console(Path(directory) / 'run.log'):
                    with self.assertRaises(io.UnsupportedOperation):
                        sys.stderr.fileno()


if __name__ == '__main__':
    unittest.main()
