"""Unit tests for merge.merge_results."""

from __future__ import annotations

import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import merge.merge_results as merge_results


def write_text(path: Path, content: str) -> None:
    """Write text file in UTF-8 encoding."""
    path.write_text(content, encoding="utf-8")


class TestMergeResults(unittest.TestCase):
    """Tests for type-based concatenation workflow."""

    def test_merge_files_for_pattern_concat(self) -> None:
        """Concatenate matching files and preserve traceability header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "2026" / "03" / "03"
            input_dir.mkdir(parents=True)
            out_file = root / "Trigger_20260303_PWM.yaml"

            write_text(input_dir / "Trigger_a_PWM.yaml", "a: 1\n")
            write_text(input_dir / "Trigger_b_PWM.yaml", "b: 2")

            code, msg = merge_results.merge_files_for_pattern(
                input_dir,
                "Trigger*PWM.yaml",
                out_file,
            )

            self.assertEqual(code, 0)
            self.assertIn("Wrote 2 files", msg)
            self.assertTrue(out_file.exists())
            text = out_file.read_text(encoding="utf-8")
            self.assertIn("# Merged: Trigger_20260303_PWM.yaml", text)
            self.assertIn("# Source files:", text)
            self.assertIn("a: 1", text)
            self.assertIn("b: 2", text)

    def test_merge_files_for_pattern_run_number_boundary_filter(self) -> None:
        """RUN10 filter should not include RUN100 files for same result type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "2026" / "03" / "03"
            input_dir.mkdir(parents=True)
            out_file = root / "Trigger_20260303_RUN10_PWM.yaml"

            write_text(input_dir / "Trigger_a_RUN10_demo_PWM.yaml", "a: 1\n")
            write_text(input_dir / "Trigger_b_RUN100_demo_PWM.yaml", "b: 2\n")

            code, msg = merge_results.merge_files_for_pattern(
                input_dir,
                "Trigger*PWM.yaml",
                out_file,
                run_number=10,
            )

            self.assertEqual(code, 0)
            self.assertIn("run_number=10", msg)
            text = out_file.read_text(encoding="utf-8")
            self.assertIn("a: 1", text)
            self.assertNotIn("b: 2", text)

    def test_merge_files_for_pattern_run_number_no_match(self) -> None:
        """Should return code 1 when requested RUN has no matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "2026" / "03" / "03"
            input_dir.mkdir(parents=True)
            out_file = root / "Trigger_20260303_RUN10_PWM.yaml"

            write_text(input_dir / "Trigger_a_RUN11_demo_PWM.yaml", "a: 1\n")

            code, msg = merge_results.merge_files_for_pattern(
                input_dir,
                "Trigger*PWM.yaml",
                out_file,
                run_number=10,
            )

            self.assertEqual(code, 1)
            self.assertIn("RUN10", msg)

    def test_merge_files_for_pattern_error_cases(self) -> None:
        """Return proper error code for missing dir and no-match cases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_file = root / "out.yaml"

            code, _ = merge_results.merge_files_for_pattern(
                root / "missing",
                "Trigger*PWM.yaml",
                out_file,
            )
            self.assertEqual(code, 2)

            empty_dir = root / "empty"
            empty_dir.mkdir()
            code, _ = merge_results.merge_files_for_pattern(
                empty_dir,
                "Trigger*PWM.yaml",
                out_file,
            )
            self.assertEqual(code, 1)

    def test_merge_all_types(self) -> None:
        """Generate merged outputs for matched/PWM/SWM patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            date_dir = outdir / "2026" / "03" / "03"
            date_dir.mkdir(parents=True)

            write_text(date_dir / "Trigger_1_matched.yaml", "m1: 1\n")
            write_text(date_dir / "Trigger_2_matched.yaml", "m2: 2\n")
            write_text(date_dir / "Trigger_1_PWM.yaml", "p1: 1\n")
            write_text(date_dir / "Trigger_2_PWM.yaml", "p2: 2\n")
            write_text(date_dir / "Trigger_1_SWM.yaml", "s1: 1\n")
            write_text(date_dir / "Trigger_2_SWM.yaml", "s2: 2\n")

            code = merge_results.merge_all_types(date_dir, "20260303", outdir)
            self.assertEqual(code, 0)
            self.assertTrue((outdir / "Trigger_20260303_matched.yaml").exists())
            self.assertTrue((outdir / "Trigger_20260303_PWM.yaml").exists())
            self.assertTrue((outdir / "Trigger_20260303_SWM.yaml").exists())

    def test_merge_all_types_with_run_number(self) -> None:
        """Run-number mode should create run-aware outputs for all result types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            date_dir = outdir / "2026" / "03" / "03"
            date_dir.mkdir(parents=True)

            write_text(date_dir / "Trigger_1_RUN65_demo_matched.yaml", "m1: 1\n")
            write_text(date_dir / "Trigger_2_RUN66_demo_matched.yaml", "m2: 2\n")
            write_text(date_dir / "Trigger_1_RUN65_demo_PWM.yaml", "p1: 1\n")
            write_text(date_dir / "Trigger_2_RUN66_demo_PWM.yaml", "p2: 2\n")
            write_text(date_dir / "Trigger_1_RUN65_demo_SWM.yaml", "s1: 1\n")
            write_text(date_dir / "Trigger_2_RUN66_demo_SWM.yaml", "s2: 2\n")

            code = merge_results.merge_all_types(date_dir, "20260303", outdir, run_number=65)
            self.assertEqual(code, 0)
            self.assertTrue((outdir / "Trigger_20260303_RUN65_matched.yaml").exists())
            self.assertTrue((outdir / "Trigger_20260303_RUN65_PWM.yaml").exists())
            self.assertTrue((outdir / "Trigger_20260303_RUN65_SWM.yaml").exists())

    def test_merge_all_types_without_run_number_logs_deprecation(self) -> None:
        """Legacy mode should print deprecation warning but keep old output names."""
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "sys.stderr"
        ) as stderr_mock:
            outdir = Path(tmpdir)
            date_dir = outdir / "2026" / "03" / "03"
            date_dir.mkdir(parents=True)

            write_text(date_dir / "Trigger_1_matched.yaml", "m1: 1\n")
            write_text(date_dir / "Trigger_1_PWM.yaml", "p1: 1\n")
            write_text(date_dir / "Trigger_1_SWM.yaml", "s1: 1\n")

            code = merge_results.merge_all_types(date_dir, "20260303", outdir)
            self.assertEqual(code, 0)
            self.assertTrue((outdir / "Trigger_20260303_matched.yaml").exists())
            self.assertTrue(stderr_mock.write.called)

    def test_main_dispatches_to_merge_all_types(self) -> None:
        """Parse args and dispatch to merge_all_types with resolved paths."""
        with mock.patch(
            "merge.merge_results.merge_all_types",
            return_value=0,
        ) as mocked_merge:
            exit_code = merge_results.main(["2026/03/03", "-o", "/tmp/out"])

        self.assertEqual(exit_code, 0)
        mocked_merge.assert_called_once()
        called_args = mocked_merge.call_args[0]
        self.assertEqual(called_args[0], Path("/tmp/out") / Path("2026/03/03"))
        self.assertEqual(called_args[1], "20260303")
        self.assertEqual(called_args[2], Path("/tmp/out"))
        self.assertIsNone(mocked_merge.call_args.kwargs["run_number"])

    def test_main_passes_run_number(self) -> None:
        """CLI run-number argument should be propagated to merge_all_types."""
        with mock.patch(
            "merge.merge_results.merge_all_types",
            return_value=0,
        ) as mocked_merge:
            exit_code = merge_results.main([
                "2026/03/03",
                "-o",
                "/tmp/out",
                "--run-number",
                "65",
            ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(mocked_merge.call_args.kwargs["run_number"], 65)

    def test_main_module_entrypoint(self) -> None:
        """Execute __main__ entrypoint and raise SystemExit."""
        old_argv = sys.argv
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                sys.argv = ["merge_results.py", "2026/03/03", "-o", tmpdir]
                with self.assertRaises(SystemExit):
                    runpy.run_path(
                        str(Path(merge_results.__file__)),
                        run_name="__main__",
                    )
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
