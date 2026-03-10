import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from simpledet.cli import main
from simpledet import __version__


class TestCli(unittest.TestCase):
    def test_main_returns_version(self):
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["--version"])

        self.assertEqual(exit_code, 0)
        self.assertIn(__version__, output.getvalue().strip())

    def test_main_prints_help_when_no_args(self):
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main([])

        self.assertEqual(exit_code, 2)
        self.assertIn("SimpleDet package bootstrap", output.getvalue())

    def test_main_forwards_check_openmmlab_to_checker(self):
        with patch("simpledet.cli._check_openmmlab", return_value=7) as patched:
            exit_code = main(["--check-openmmlab"])

        self.assertEqual(exit_code, 7)
        patched.assert_called_once()

    def test_main_forwards_check_runtime_alias_to_checker(self):
        with patch("simpledet.cli._check_openmmlab", return_value=3) as patched:
            exit_code = main(["--check-runtime"])

        self.assertEqual(exit_code, 3)
        patched.assert_called_once()
