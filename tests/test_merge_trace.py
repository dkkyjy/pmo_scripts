"""Unit tests for merge.merge_trace."""

from __future__ import annotations

import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import yaml

import merge.merge_trace as merge_trace


def write_yaml(path: Path, data: Dict[str, Dict[str, Any]]) -> None:
    """Write dictionary to YAML file."""
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def read_merged_yaml_without_header(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read merged YAML and strip leading comment header lines."""
    raw_text = path.read_text(encoding="utf-8")
    body = "\n".join(
        line for line in raw_text.splitlines() if not line.startswith("#")
    ).strip()
    if not body:
        return {}
    loaded = yaml.safe_load(body)
    if not isinstance(loaded, dict):
        return {}
    return loaded


class TestMergeTrace(unittest.TestCase):
    """Tests for merge_trace multi-trigger merge behavior."""

    def test_merge_event_payload_keeps_multiple_samples_per_du(self) -> None:
        """Should concatenate time/signal values for duplicated DU keys."""
        base = {
            "event_number": 10,
            "du_id": ["101", "102"],
            "time": {"101": [100], "102": 200},
            "signal": {"101": [7], "102": 8},
        }
        incoming = {
            "event_number": 10,
            "du_id": ["101", "103"],
            "time": {"101": [101, 102], "102": 201, "103": [300]},
            "signal": {"101": [9], "102": 10, "103": [11]},
        }

        merged = merge_trace.merge_event_payload(base, incoming)

        self.assertEqual(merged["time"]["101"], [100, 101, 102])
        self.assertEqual(merged["time"]["102"], [200, 201])
        self.assertEqual(merged["time"]["103"], [300])
        self.assertEqual(merged["signal"]["101"], [7, 9])
        self.assertEqual(merged["signal"]["102"], [8, 10])
        self.assertEqual(merged["signal"]["103"], [11])
        self.assertEqual(merged["du_id"], ["101", "102", "103"])

    def test_merge_yaml_by_event_number_keeps_multiple_samples(self) -> None:
        """Should keep merged list samples when duplicate event_number appears."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            file_a = root / "Trigger_a_XY.yaml"
            file_b = root / "Trigger_b_XY.yaml"

            write_yaml(
                file_a,
                {
                    "k1": {
                        "event_number": 100,
                        "du_id": ["101"],
                        "time": {"101": [100]},
                        "signal": {"101": [7]},
                    }
                },
            )
            write_yaml(
                file_b,
                {
                    "k2": {
                        "event_number": 100,
                        "du_id": ["101", "102"],
                        "time": {"101": [101], "102": [200]},
                        "signal": {"101": [8], "102": [9]},
                    }
                },
            )

            merged = merge_trace.merge_yaml_by_event_number([file_a, file_b])
            self.assertIn("100", merged)
            self.assertEqual(merged["100"]["time"]["101"], [100, 101])
            self.assertEqual(merged["100"]["time"]["102"], [200])
            self.assertEqual(merged["100"]["signal"]["101"], [7, 8])
            self.assertEqual(merged["100"]["signal"]["102"], [9])

    def test_merge_files_for_pattern_xy(self) -> None:
        """Should write merged XY output while preserving merged DU samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_file = root / "Trigger_20260306_XY_merged.yaml"

            write_yaml(
                root / "Trigger_1_XY.yaml",
                {
                    "a": {
                        "event_number": 1,
                        "du_id": ["1"],
                        "time": {"1": [10]},
                        "signal": {"1": [20]},
                    }
                },
            )
            write_yaml(
                root / "Trigger_2_XY.yaml",
                {
                    "b": {
                        "event_number": 1,
                        "du_id": ["1"],
                        "time": {"1": [11]},
                        "signal": {"1": [21]},
                    }
                },
            )

            code, _ = merge_trace.merge_files_for_pattern(
                root,
                "Trigger*_XY.yaml",
                out_file,
            )
            self.assertEqual(code, 0)
            body = read_merged_yaml_without_header(out_file)
            self.assertEqual(body["1"]["time"]["1"], [10, 11])
            self.assertEqual(body["1"]["signal"]["1"], [20, 21])

    def test_merge_files_for_pattern_run_number_boundary_filter(self) -> None:
        """RUN10 filter should not include RUN100 files for same trace type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_file = root / "Trigger_20260306_RUN10_XY_merged.yaml"

            write_yaml(
                root / "Trigger_20260306_RUN10_demo_XY.yaml",
                {
                    "a": {
                        "event_number": 10,
                        "du_id": ["10"],
                        "time": {"10": [10]},
                    }
                },
            )
            write_yaml(
                root / "Trigger_20260306_RUN100_demo_XY.yaml",
                {
                    "b": {
                        "event_number": 100,
                        "du_id": ["100"],
                        "time": {"100": [100]},
                    }
                },
            )

            code, msg = merge_trace.merge_files_for_pattern(
                root,
                "Trigger*_XY.yaml",
                out_file,
                run_number=10,
            )

            self.assertEqual(code, 0)
            self.assertIn("run_number=10", msg)
            body = read_merged_yaml_without_header(out_file)
            self.assertEqual(sorted(body.keys()), ["10"])

    def test_merge_files_for_pattern_run_number_no_match(self) -> None:
        """Should return code 1 when requested RUN has no matching files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_file = root / "Trigger_20260306_RUN10_XY_merged.yaml"

            write_yaml(
                root / "Trigger_20260306_RUN11_demo_XY.yaml",
                {
                    "a": {
                        "event_number": 11,
                        "du_id": ["11"],
                        "time": {"11": [11]},
                    }
                },
            )

            code, msg = merge_trace.merge_files_for_pattern(
                root,
                "Trigger*_XY.yaml",
                out_file,
                run_number=10,
            )

            self.assertEqual(code, 1)
            self.assertIn("RUN10", msg)

    def test_merge_event_payload_initializes_missing_maps_and_lists(self) -> None:
        """Should initialize missing dict/list fields before merging."""
        base = {
            "event_number": 10,
            "time": None,
            "signal": None,
            "du_id": None,
        }
        incoming = {
            "event_number": 10,
            "time": {"1": 1},
            "signal": {"1": 2},
            "du_id": ["1", "2", "1"],
        }

        merged = merge_trace.merge_event_payload(base, incoming)
        self.assertEqual(merged["time"], {"1": 1})
        self.assertEqual(merged["signal"], {"1": 2})
        self.assertEqual(merged["du_id"], ["1", "2"])

    def test_merge_event_payload_sets_missing_scalar_field(self) -> None:
        """Should set scalar field when absent in base payload."""
        base = {"event_number": 20}
        incoming = {"event_number": 20, "index": 7}
        merged = merge_trace.merge_event_payload(base, incoming)
        self.assertEqual(merged["index"], 7)

    def test_load_yaml_dict_handles_empty_and_nondict(self) -> None:
        """Should return empty dict for empty YAML and non-dict top-level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            empty_file = root / "empty.yaml"
            list_file = root / "list.yaml"

            empty_file.write_text("", encoding="utf-8")
            list_file.write_text("- 1\n- 2\n", encoding="utf-8")

            self.assertEqual(merge_trace.load_yaml_dict(empty_file), {})
            self.assertEqual(merge_trace.load_yaml_dict(list_file), {})

    def test_merge_yaml_by_event_number_skips_invalid_entries(self) -> None:
        """Should skip non-dict payload and payload without event_number."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            file_a = root / "Trigger_a_XY.yaml"
            write_yaml(
                file_a,
                {
                    "non_dict": [1, 2],
                    "no_event": {"du_id": ["1"], "time": {"1": 11}},
                    "ok": {
                        "event_number": 9,
                        "du_id": ["9"],
                        "time": {"9": 99},
                    },
                },
            )

            merged = merge_trace.merge_yaml_by_event_number([file_a])
            self.assertEqual(list(merged.keys()), ["9"])

    def test_merge_yaml_by_event_number_without_duplicates(self) -> None:
        """Should keep single records and cover no-duplicate branch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            file_a = root / "Trigger_a_F.yaml"
            write_yaml(
                file_a,
                {
                    "k1": {
                        "event_number": 1,
                        "du_id": ["1"],
                        "time": {"1": [10]},
                    }
                },
            )

            merged = merge_trace.merge_yaml_by_event_number([file_a])
            self.assertEqual(list(merged.keys()), ["1"])

    def test_merge_files_for_pattern_error_cases(self) -> None:
        """Should return error codes for missing directory or no files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_file = root / "merged.yaml"

            code, _ = merge_trace.merge_files_for_pattern(
                root / "not_exists",
                "Trigger*_XY.yaml",
                out_file,
            )
            self.assertEqual(code, 2)

            empty_dir = root / "empty"
            empty_dir.mkdir()
            code, _ = merge_trace.merge_files_for_pattern(
                empty_dir,
                "Trigger*_XY.yaml",
                out_file,
            )
            self.assertEqual(code, 1)

    def test_parse_date_dir(self) -> None:
        """Should parse date path and produce yyyymmdd string."""
        date_path, ymd = merge_trace.parse_date_dir("2026/03/03")
        self.assertEqual(date_path, Path("2026/03/03"))
        self.assertEqual(ymd, "20260303")

    def test_merge_trigger_files_all_types(self) -> None:
        """Should write merged outputs for all configured type patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            date_dir = outdir / "2026" / "03" / "03"
            date_dir.mkdir(parents=True)

            for suffix in ["F", "X", "Y", "Z", "XY"]:
                write_yaml(
                    date_dir / f"Trigger_a_{suffix}.yaml",
                    {
                        "a": {
                            "event_number": 1,
                            "du_id": ["1"],
                            "time": {"1": [10]},
                            "signal": {"1": [20]},
                        }
                    },
                )
                write_yaml(
                    date_dir / f"Trigger_b_{suffix}.yaml",
                    {
                        "b": {
                            "event_number": 1,
                            "du_id": ["2"],
                            "time": {"2": [11]},
                            "signal": {"2": [21]},
                        }
                    },
                )

            code = merge_trace.merge_trigger_files(date_dir, "20260303", outdir)
            self.assertEqual(code, 0)

            merged_xy = outdir / "Trigger_20260303_XY_merged.yaml"
            self.assertTrue(merged_xy.exists())
            body = read_merged_yaml_without_header(merged_xy)
            self.assertEqual(body["1"]["du_id"], ["1", "2"])

    def test_merge_trigger_files_with_run_number_all_types(self) -> None:
        """Run-number mode should generate per-type outputs with RUN segment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            date_dir = outdir / "2026" / "03" / "03"
            date_dir.mkdir(parents=True)

            for suffix in ["F", "X", "Y", "Z", "XY"]:
                write_yaml(
                    date_dir / f"Trigger_20260303_RUN65_demo_{suffix}.yaml",
                    {
                        "a": {
                            "event_number": 1,
                            "du_id": ["1"],
                            "time": {"1": [10]},
                        }
                    },
                )
                write_yaml(
                    date_dir / f"Trigger_20260303_RUN66_demo_{suffix}.yaml",
                    {
                        "b": {
                            "event_number": 2,
                            "du_id": ["2"],
                            "time": {"2": [11]},
                        }
                    },
                )

            code = merge_trace.merge_trigger_files(date_dir, "20260303", outdir, run_number=65)
            self.assertEqual(code, 0)

            merged_xy = outdir / "Trigger_20260303_RUN65_XY_merged.yaml"
            self.assertTrue(merged_xy.exists())
            body = read_merged_yaml_without_header(merged_xy)
            self.assertEqual(sorted(body.keys()), ["1"])

    def test_merge_trigger_files_without_run_number_logs_deprecation_warning(self) -> None:
        """Legacy mode should log deprecation warning and keep old naming."""
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "merge.merge_trace.logger.warning"
        ) as warning_mock:
            outdir = Path(tmpdir)
            date_dir = outdir / "2026" / "03" / "03"
            date_dir.mkdir(parents=True)

            write_yaml(
                date_dir / "Trigger_a_F.yaml",
                {
                    "a": {
                        "event_number": 1,
                        "du_id": ["1"],
                        "time": {"1": [10]},
                    }
                },
            )

            code = merge_trace.merge_trigger_files(date_dir, "20260303", outdir)
            self.assertEqual(code, 1)
            warning_mock.assert_called_once()

    def test_merge_trigger_files_stops_on_error(self) -> None:
        """Should stop and return non-zero when one type merge fails."""
        with mock.patch(
            "merge.merge_trace.merge_files_for_pattern",
            side_effect=[(0, "ok"), (1, "fail")],
        ):
            code = merge_trace.merge_trigger_files(
                Path("2026/03/03"),
                "20260303",
                Path("/tmp/out"),
            )
        self.assertEqual(code, 1)

    def test_main_dispatches_to_merge_trigger_files(self) -> None:
        """Should parse args and call merge_trigger_files with resolved paths."""
        with mock.patch(
            "merge.merge_trace.merge_trigger_files",
            return_value=0,
        ) as mocked:
            exit_code = merge_trace.main(["2026/03/03", "-o", "/tmp/out"])

        self.assertEqual(exit_code, 0)
        mocked.assert_called_once()
        called_args = mocked.call_args[0]
        self.assertEqual(called_args[0], Path("/tmp/out") / Path("2026/03/03"))
        self.assertEqual(called_args[1], "20260303")
        self.assertEqual(called_args[2], Path("/tmp/out"))
        self.assertIsNone(mocked.call_args.kwargs["run_number"])

    def test_main_passes_run_number(self) -> None:
        """CLI run-number argument should be propagated to merge_trigger_files."""
        with mock.patch(
            "merge.merge_trace.merge_trigger_files",
            return_value=0,
        ) as mocked:
            exit_code = merge_trace.main([
                "2026/03/03",
                "-o",
                "/tmp/out",
                "--run-number",
                "65",
            ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(mocked.call_args.kwargs["run_number"], 65)

    def test_main_module_entrypoint(self) -> None:
        """Should execute __main__ entrypoint and raise SystemExit."""
        old_argv = sys.argv
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                sys.argv = ["merge_trace.py", "2026/03/03", "-o", tmpdir]
                with self.assertRaises(SystemExit):
                    runpy.run_path(
                        str(Path(merge_trace.__file__)),
                        run_name="__main__",
                    )
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
