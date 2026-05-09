"""Unit tests for merge.merge_header."""

from __future__ import annotations

import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import yaml

import merge.merge_header as merge


def write_yaml(path: Path, data: Dict[str, Dict[str, Any]]) -> None:
    """Write dictionary to YAML file."""
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def read_merged_yaml_without_header(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read merged YAML and strip leading comment header lines."""
    raw_text = path.read_text(encoding="utf-8")
    body = "\n".join(line for line in raw_text.splitlines() if not line.startswith("#")).strip()
    if not body:
        return {}
    loaded = yaml.safe_load(body)
    if not isinstance(loaded, dict):
        return {}
    return loaded


class TestMerge(unittest.TestCase):
    """Tests for merge helper and event-number merge workflow."""

    def test_merge_event_payload(self) -> None:
        """Should merge time/signal maps and deduplicate du_id."""
        base = {
            "event_number": 10,
            "du_id": ["1014", "1018"],
            "time": {"1014": 1, "1018": 2},
            "signal": {"1014": 11, "1018": 12},
            "run_number": 1,
        }
        incoming = {
            "event_number": 10,
            "du_id": ["1018", "1024"],
            "time": {"1024": 3},
            "signal": {"1024": 13},
            "index": 5,
        }

        merged = merge.merge_event_payload(base, incoming)

        self.assertEqual(merged["du_id"], ["1014", "1018", "1024"])
        self.assertEqual(merged["time"], {"1014": 1, "1018": 2, "1024": 3})
        self.assertEqual(
            merged["signal"],
            {"1014": 11, "1018": 12, "1024": 13},
        )
        self.assertEqual(merged["run_number"], 1)
        self.assertEqual(merged["index"], 5)

    def test_merge_event_payload_when_base_du_id_missing(self) -> None:
        """Should handle incoming du_id when base has no du_id list."""
        base = {
            "event_number": 20,
            "du_id": None,
            "time": {},
        }
        incoming = {
            "event_number": 20,
            "du_id": ["2001", "2002"],
            "time": {"2001": 1},
        }

        merged = merge.merge_event_payload(base, incoming)
        self.assertEqual(merged["du_id"], ["2001", "2002"])

    def test_merge_event_payload_when_base_time_not_dict(self) -> None:
        """Should initialize time map when base payload has non-dict time."""
        base = {
            "event_number": 22,
            "time": None,
        }
        incoming = {
            "event_number": 22,
            "time": {"2001": 456},
        }

        merged = merge.merge_event_payload(base, incoming)
        self.assertEqual(merged["time"], {"2001": 456})

    def test_merge_sample_map_concatenates_scalars(self) -> None:
        """Should concatenate scalars into a list when merging."""
        base = {"101": 1.0}
        incoming = {"101": 2.0}
        merged = merge.merge_sample_map(base, incoming)
        self.assertEqual(merged["101"], [1.0, 2.0])

    def test_merge_sample_map_concatenates_lists(self) -> None:
        """Should concatenate lists when merging."""
        base = {"101": [1.0, 2.0]}
        incoming = {"101": [3.0, 4.0]}
        merged = merge.merge_sample_map(base, incoming)
        self.assertEqual(merged["101"], [1.0, 2.0, 3.0, 4.0])

    def test_merge_event_payload_keeps_existing_non_none(self) -> None:
        """Should keep existing field value if it's already set and incoming is provided."""
        base = {"event_number": 50, "custom": "old"}
        incoming = {"event_number": 50, "custom": "new"}
        merged = merge.merge_event_payload(base, incoming)
        self.assertEqual(merged["custom"], "old")

    def test_merge_event_payload_sets_if_missing_or_none(self) -> None:
        """Should set field from incoming if it's missing or None in base."""
        base = {"event_number": 51, "custom": None}
        incoming = {"event_number": 51, "custom": "set"}
        merged = merge.merge_event_payload(base, incoming)
        self.assertEqual(merged["custom"], "set")

    def test_load_yaml_dict_non_dict(self) -> None:
        """load_yaml_dict should return empty dict if YAML is not a dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "list.yaml"
            path.write_text("- item1\n- item2", encoding="utf-8")
            self.assertEqual(merge.load_yaml_dict(path), {})

    def test_merge_event_payload_preserves_file_and_index_values(self) -> None:
        """Should preserve both existing and incoming values for file/index."""
        base = {
            "event_number": 31,
            "file": "a.root",
            "index": 10,
        }
        incoming = {
            "event_number": 31,
            "file": "b.root",
            "index": 11,
        }

        merged = merge.merge_event_payload(base, incoming)

        self.assertEqual(merged["file"], ["a.root", "b.root"])
        self.assertEqual(merged["index"], [10, 11])

    def test_load_yaml_dict_handles_none_and_nondict(self) -> None:
        """Should return empty dict for empty YAML or non-dict YAML top-level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            empty_file = root / "empty.yaml"
            list_file = root / "list.yaml"

            empty_file.write_text("", encoding="utf-8")
            list_file.write_text("- 1\n- 2\n", encoding="utf-8")

            self.assertEqual(merge.load_yaml_dict(empty_file), {})
            self.assertEqual(merge.load_yaml_dict(list_file), {})

    def test_merge_yaml_by_event_number(self) -> None:
        """Should merge records sharing same event_number across files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            file_a = root / "Trigger_a_matched.yaml"
            file_b = root / "Trigger_b_matched.yaml"

            write_yaml(
                file_a,
                {
                    "k1": {"event_number": 100, "du_id": ["1014"], "time": {"1014": 1}},
                    "k2": {"event_number": 101, "du_id": ["1018"], "time": {"1018": 2}},
                },
            )
            write_yaml(
                file_b,
                {
                    "k3": {
                        "event_number": 100,
                        "du_id": ["1024"],
                        "time": {"1024": 3},
                    }
                },
            )

            merged = merge.merge_yaml_by_event_number([file_a, file_b])

            self.assertEqual(sorted(merged.keys()), ["100", "101"])
            self.assertEqual(merged["100"]["du_id"], ["1014", "1024"])
            self.assertEqual(merged["100"]["time"], {"1014": 1, "1024": 3})

    def test_merge_yaml_by_event_number_merges_adjacent_three_files(self) -> None:
        """Three adjacent files with same event_number should merge in sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            file_a = root / "Trigger_a.yaml"
            file_b = root / "Trigger_b.yaml"
            file_c = root / "Trigger_c.yaml"

            write_yaml(
                file_a,
                {"a": {"event_number": 200, "du_id": ["1"], "time": {"1": 1}}},
            )
            write_yaml(
                file_b,
                {"b": {"event_number": 200, "du_id": ["2"], "time": {"2": 2}}},
            )
            write_yaml(
                file_c,
                {"c": {"event_number": 200, "du_id": ["3"], "time": {"3": 3}}},
            )

            merged = merge.merge_yaml_by_event_number([file_a, file_b, file_c])

            self.assertEqual(sorted(merged.keys()), ["200"])
            self.assertEqual(merged["200"]["du_id"], ["1", "2", "3"])
            self.assertEqual(merged["200"]["time"], {"1": 1, "2": 2, "3": 3})

    def test_merge_yaml_by_event_number_does_not_merge_non_adjacent_duplicate(self) -> None:
        """Non-adjacent duplicate event_number should not merge across a gap file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            file_a = root / "Trigger_a.yaml"
            file_b = root / "Trigger_b.yaml"
            file_c = root / "Trigger_c.yaml"

            write_yaml(
                file_a,
                {"a": {"event_number": 300, "du_id": ["10"], "time": {"10": 10}}},
            )
            write_yaml(
                file_b,
                {"b": {"event_number": 999, "du_id": ["99"], "time": {"99": 99}}},
            )
            write_yaml(
                file_c,
                {"c": {"event_number": 300, "du_id": ["30"], "time": {"30": 30}}},
            )

            merged = merge.merge_yaml_by_event_number([file_a, file_b, file_c])

            self.assertEqual(sorted(merged.keys()), ["300", "999"])
            self.assertEqual(merged["300"]["du_id"], ["30"])
            self.assertEqual(merged["300"]["time"], {"30": 30})

    def test_merge_yaml_by_event_number_single_file_passthrough(self) -> None:
        """Single file input should keep payload content unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            file_a = root / "Trigger_single.yaml"

            write_yaml(
                file_a,
                {
                    "x": {
                        "event_number": 77,
                        "du_id": ["701", "702"],
                        "time": {"701": 7, "702": 8},
                        "signal": {"701": 70, "702": 80},
                    }
                },
            )

            merged = merge.merge_yaml_by_event_number([file_a])

            self.assertEqual(sorted(merged.keys()), ["77"])
            self.assertEqual(merged["77"]["du_id"], ["701", "702"])
            self.assertEqual(merged["77"]["time"], {"701": 7, "702": 8})
            self.assertEqual(merged["77"]["signal"], {"701": 70, "702": 80})

    def test_merge_yaml_by_event_number_skips_invalid_entries(self) -> None:
        """Should skip non-dict payload and payload without event_number."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            file_a = root / "Trigger_a_matched.yaml"
            write_yaml(
                file_a,
                {
                    "non_dict": [1, 2],
                    "no_event": {"du_id": ["1"], "time": {"1": 11}},
                    "ok": {"event_number": 9, "du_id": ["9"], "time": {"9": 99}},
                },
            )

            merged = merge.merge_yaml_by_event_number([file_a])
            self.assertEqual(list(merged.keys()), ["9"])

    def test_merge_files_for_pattern_event_number(self) -> None:
        """Should write merged output file in fixed event-number mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "2026" / "03" / "03"
            input_dir.mkdir(parents=True)
            out_file = root / "merged.yaml"

            write_yaml(
                input_dir / "Trigger_a_matched.yaml",
                {"a": {"event_number": 1, "du_id": ["1"], "time": {"1": 1}}},
            )
            write_yaml(
                input_dir / "Trigger_b_matched.yaml",
                {"b": {"event_number": 1, "du_id": ["2"], "time": {"2": 2}}},
            )

            code, msg = merge.merge_files_for_pattern(
                input_dir,
                "Trigger*.yaml",
                out_file,
            )

            self.assertEqual(code, 0)
            self.assertIn("mode=event-number", msg)
            self.assertTrue(out_file.exists())

            data = read_merged_yaml_without_header(out_file)
            self.assertIn("1", data)
            self.assertEqual(data["1"]["du_id"], ["1", "2"])

    def test_merge_files_for_pattern_run_number_boundary_filter(self) -> None:
        """RUN10 filtering should not match RUN100 files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "2026" / "03" / "03"
            input_dir.mkdir(parents=True)
            out_file = root / "merged.yaml"

            write_yaml(
                input_dir / "Trigger_x_RUN10_A.yaml",
                {"a": {"event_number": 10, "du_id": ["10"], "time": {"10": 1}}},
            )
            write_yaml(
                input_dir / "Trigger_x_RUN100_A.yaml",
                {"b": {"event_number": 100, "du_id": ["100"], "time": {"100": 2}}},
            )

            code, msg = merge.merge_files_for_pattern(
                input_dir,
                "Trigger*.yaml",
                out_file,
                run_number=10,
            )

            self.assertEqual(code, 0)
            self.assertIn("run_number=10", msg)
            data = read_merged_yaml_without_header(out_file)
            self.assertEqual(sorted(data.keys()), ["10"])

    def test_merge_files_for_pattern_run_number_no_match(self) -> None:
        """Should return code 1 when no files match the requested RUN number."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "2026" / "03" / "03"
            input_dir.mkdir(parents=True)
            out_file = root / "merged.yaml"

            write_yaml(
                input_dir / "Trigger_x_RUN11_A.yaml",
                {"a": {"event_number": 11, "du_id": ["11"], "time": {"11": 1}}},
            )

            code, msg = merge.merge_files_for_pattern(
                input_dir,
                "Trigger*.yaml",
                out_file,
                run_number=10,
            )

            self.assertEqual(code, 1)
            self.assertIn("RUN10", msg)

    def test_merge_files_for_pattern_error_cases(self) -> None:
        """Should return correct error code when directory/files are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_file = root / "merged.yaml"

            code, _ = merge.merge_files_for_pattern(
                root / "not_exists",
                "Trigger*.yaml",
                out_file,
            )
            self.assertEqual(code, 2)

            empty_dir = root / "empty"
            empty_dir.mkdir()
            code, _ = merge.merge_files_for_pattern(
                empty_dir,
                "Trigger*.yaml",
                out_file,
            )
            self.assertEqual(code, 1)

    def test_parse_date_dir(self) -> None:
        """Should parse date directory into path and yyyymmdd."""
        date_path, ymd = merge.parse_date_dir("2026/03/03")
        self.assertEqual(date_path, Path("2026/03/03"))
        self.assertEqual(ymd, "20260303")

    def test_merge_trigger_files(self) -> None:
        """Should merge Trigger*.yaml into Trigger_yyyymmdd_merged.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            date_dir = outdir / "2026" / "03" / "03"
            date_dir.mkdir(parents=True)

            write_yaml(
                date_dir / "Trigger_a.yaml",
                {"a": {"event_number": 1, "du_id": ["1"], "time": {"1": 1}}},
            )
            write_yaml(
                date_dir / "Trigger_b.yaml",
                {"b": {"event_number": 1, "du_id": ["2"], "time": {"2": 2}}},
            )

            code = merge.merge_trigger_files(date_dir, "20260303", outdir)
            self.assertEqual(code, 0)

            merged_out = outdir / "Trigger_20260303_merged.yaml"
            self.assertTrue(merged_out.exists())
            body = read_merged_yaml_without_header(merged_out)
            self.assertIn("1", body)
            self.assertEqual(body["1"]["du_id"], ["1", "2"])

    def test_merge_trigger_files_with_run_number_uses_run_output_name(self) -> None:
        """Run-number mode should write output file including RUN segment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            date_dir = outdir / "2026" / "03" / "03"
            date_dir.mkdir(parents=True)

            write_yaml(
                date_dir / "Trigger_a_RUN65_A.yaml",
                {"a": {"event_number": 1, "du_id": ["1"], "time": {"1": 1}}},
            )
            write_yaml(
                date_dir / "Trigger_b_RUN66_A.yaml",
                {"b": {"event_number": 2, "du_id": ["2"], "time": {"2": 2}}},
            )

            code = merge.merge_trigger_files(date_dir, "20260303", outdir, run_number=65)
            self.assertEqual(code, 0)

            merged_out = outdir / "Trigger_20260303_RUN65_merged.yaml"
            self.assertTrue(merged_out.exists())
            body = read_merged_yaml_without_header(merged_out)
            self.assertEqual(sorted(body.keys()), ["1"])

    def test_merge_trigger_files_without_run_number_logs_deprecation_warning(self) -> None:
        """Legacy mode should keep old naming and emit deprecation warning."""
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "merge.merge_header.logger.warning"
        ) as warning_mock:
            outdir = Path(tmpdir)
            date_dir = outdir / "2026" / "03" / "03"
            date_dir.mkdir(parents=True)

            write_yaml(
                date_dir / "Trigger_a.yaml",
                {"a": {"event_number": 1, "du_id": ["1"], "time": {"1": 1}}},
            )

            code = merge.merge_trigger_files(date_dir, "20260303", outdir)
            self.assertEqual(code, 0)
            self.assertTrue((outdir / "Trigger_20260303_merged.yaml").exists())
            warning_mock.assert_called_once()

    def test_main_calls_merge_all_types(self) -> None:
        """Should parse args and dispatch to merge_trigger_files with expected values."""
        with mock.patch(
            "merge.merge_header.merge_trigger_files",
            return_value=0,
        ) as mocked_merge:
            exit_code = merge.main(["2026/03/03", "-o", "/tmp/out"])

        self.assertEqual(exit_code, 0)
        mocked_merge.assert_called_once()
        called_args = mocked_merge.call_args[0]
        self.assertEqual(called_args[0], Path("/tmp/out") / Path("2026/03/03"))
        self.assertEqual(called_args[1], "20260303")
        self.assertEqual(called_args[2], Path("/tmp/out"))
        self.assertIsNone(mocked_merge.call_args.kwargs["run_number"])

    def test_main_passes_run_number(self) -> None:
        """CLI run-number argument should be propagated to merge_trigger_files."""
        with mock.patch(
            "merge.merge_header.merge_trigger_files",
            return_value=0,
        ) as mocked_merge:
            exit_code = merge.main([
                "2026/03/03",
                "-o",
                "/tmp/out",
                "--run-number",
                "65",
            ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(mocked_merge.call_args.kwargs["run_number"], 65)

    def test_main_module_entrypoint(self) -> None:
        """Should execute __main__ entrypoint (raise SystemExit from main)."""
        old_argv = sys.argv
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                sys.argv = ["merge_header.py", "2026/03/03", "-o", tmpdir]
                with self.assertRaises(SystemExit):
                    runpy.run_path(str(Path(merge.__file__)), run_name="__main__")
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
