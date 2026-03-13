"""Unit tests for stats_lookback.py."""

from __future__ import annotations

import csv
import runpy
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt

import yaml

import stats.stats_lookback as sadp


def make_data() -> dict:
    """Create sample Trigger-like YAML payloads."""
    return {
        "100": {
            "event_number": 100,
            "gps_time": 10,
            "index": 0,
            "datetime": "2026-02-14T12:00:00",
            "time": {"1": 100.0, "2": 140.0, "3": 180.0},
        },
        "101": {
            "event_number": 101,
            "gps_time": 11,
            "index": 1,
            "datetime": "2026-02-14T12:00:01",
            "time": {"1": 110.0, "2": 150.0, "4": 190.0},
        },
        "102": {
            "event_number": 102,
            "gps_time": 12,
            "index": 2,
            "datetime": "2026-02-14T12:00:02",
            "time": {"1": 120.0, "3": 200.0, "4": 260.0},
        },
    }


class TestStatsAdjacentDuPair(unittest.TestCase):
    """Tests for previous-N-event common DU-pair delta workflow."""

    def test_sort_du_id_key_mixed(self) -> None:
        """Numeric DU IDs should sort before non-numeric IDs."""
        keys = ["10", "A1", "2"]
        sorted_keys = sorted(keys, key=sadp.sort_du_id_key)
        self.assertEqual(sorted_keys, ["2", "10", "A1"])

    def test_read_yaml_events_empty_and_invalid(self) -> None:
        """Should handle empty YAML and reject non-dict top-level objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            empty_yaml = root / "empty.yaml"
            empty_yaml.write_text("", encoding="utf-8")
            self.assertEqual(sadp.read_yaml_events(empty_yaml), {})

            invalid_yaml = root / "invalid.yaml"
            invalid_yaml.write_text(yaml.safe_dump([1, 2, 3]), encoding="utf-8")
            with self.assertRaises(ValueError):
                sadp.read_yaml_events(invalid_yaml)

    def test_parse_du_ns_map_invalid_values(self) -> None:
        """Should skip invalid time values and non-dict time payloads."""
        parsed = sadp.parse_du_ns_map({"time": {"1": 100, "2": "bad", "3": 3.5}})
        self.assertEqual(parsed, {"1": 100.0, "3": 3.5})
        self.assertEqual(sadp.parse_du_ns_map({"time": None}), {})

        parsed_list = sadp.parse_du_ns_map(
            {"time": {"1": [100, "130"], "2": ["bad"]}}
        )
        self.assertEqual(parsed_list, {"1": 130.0})

    def test_build_event_records_sort_and_parse(self) -> None:
        """Should parse records and keep sort order by gps/index/event."""
        records = sadp.build_event_records(make_data())
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["event_number"], 100)
        self.assertEqual(records[1]["event_number"], 101)
        self.assertEqual(records[2]["event_number"], 102)
        self.assertEqual(records[1]["event_datetime"], "2026-02-14T12:00:01")

    def test_build_event_records_invalid_fields(self) -> None:
        """Should skip/normalize events with malformed gps/index/event fields."""
        data = {
            "good": {
                "event_number": "11",
                "gps_time": "20",
                "index": "2",
                "time": {"1": 1},
            },
            "bad_gps": {
                "event_number": 12,
                "gps_time": "x",
                "index": 3,
                "time": {"1": 1},
            },
            "negative_gps": {
                "event_number": 13,
                "gps_time": -1,
                "index": 4,
                "time": {"1": 1},
            },
            "bad_event_index": {
                "event_number": "bad",
                "gps_time": 21,
                "index": "bad",
                "time": {"1": 2},
            },
        }
        records = sadp.build_event_records(data)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["event_number"], 11)
        self.assertEqual(records[1]["event_number"], -1)
        self.assertEqual(records[1]["index"], -1)
        self.assertEqual(records[1]["event_datetime"], "")

    def test_build_pair_delta_map_empty(self) -> None:
        """Empty DU map should return empty pair map."""
        self.assertEqual(sadp.build_pair_delta_map({}), {})

    def test_build_pair_delta_map_with_offsets(self) -> None:
        """Offset correction should be applied before pair-delta calculation."""
        du_ns_map = {"1": 100.0, "2": 140.0}
        offset_map = {"1": 10.0, "2": -10.0}
        pair_map = sadp.build_pair_delta_map(du_ns_map, offset_map)
        self.assertAlmostEqual(pair_map[("1", "2")], 60.0)

    def test_adjacent_rows_for_short_input(self) -> None:
        """Less than two events should produce no rows."""
        self.assertEqual(sadp.build_adjacent_common_pair_rows([]), [])

    def test_default_lookback_compares_previous_n_events(self) -> None:
        """Default lookback should compare each event with up to 10 previous."""
        records = sadp.build_event_records(make_data())
        rows = sadp.build_adjacent_common_pair_rows(records)

        self.assertEqual(len(rows), 3)
        actual = {
            (
                row["prev_event_number"],
                row["curr_event_number"],
                row["du_a"],
                row["du_b"],
                row["prev_delta_ns"],
                row["curr_delta_ns"],
                row["adjacent_delta_ns"],
                row["abs_adjacent_delta_ns"],
            )
            for row in rows
        }
        expected = {
            (100, 101, "1", "2", 40.0, 40.0, 0.0, 0.0),
            (101, 102, "1", "4", 80.0, 140.0, 60.0, 60.0),
            (100, 102, "1", "3", 80.0, 80.0, 0.0, 0.0),
        }
        self.assertSetEqual(actual, expected)

    def test_lookback_one_keeps_adjacent_only(self) -> None:
        """Lookback=1 should keep adjacent comparisons only."""
        records = sadp.build_event_records(make_data())
        rows = sadp.build_adjacent_common_pair_rows(records, lookback=1)
        self.assertEqual(len(rows), 2)

    def test_build_shared_pair_count_rows(self) -> None:
        """Should build shared DU-pair count rows for lookback comparisons."""
        records = sadp.build_event_records(make_data())
        count_rows = sadp.build_shared_pair_count_rows(records, lookback=10)
        self.assertEqual(len(count_rows), 3)

        actual = {
            (
                row["prev_event_number"],
                row["curr_event_number"],
                row["shared_pair_count"],
            )
            for row in count_rows
        }
        expected = {
            (100, 101, 1),
            (101, 102, 1),
            (100, 102, 1),
        }
        self.assertSetEqual(actual, expected)

    def test_build_shared_du_count_rows(self) -> None:
        """Should build shared DU-id count rows for lookback comparisons."""
        records = sadp.build_event_records(make_data())
        count_rows = sadp.build_shared_du_count_rows(records, lookback=10)
        self.assertEqual(len(count_rows), 3)

        actual = {
            (
                row["prev_event_number"],
                row["curr_event_number"],
                row["shared_du_count"],
            )
            for row in count_rows
        }
        expected = {
            (100, 101, 2),
            (101, 102, 2),
            (100, 102, 2),
        }
        self.assertSetEqual(actual, expected)

    def test_derive_shared_pair_count_rows_from_adjacent_rows(self) -> None:
        """Should derive per-comparison pair counts from adjacent pair rows."""
        rows = sadp.build_adjacent_common_pair_rows(sadp.build_event_records(make_data()))
        count_rows = sadp.derive_shared_pair_count_rows_from_adjacent_rows(rows)
        self.assertEqual(len(count_rows), 3)
        self.assertTrue(all(row["shared_pair_count"] == 1 for row in count_rows))

    def test_lookback_non_positive_is_normalized(self) -> None:
        """Non-positive lookback values should be normalized to 1."""
        records = sadp.build_event_records(make_data())
        rows_zero = sadp.build_adjacent_common_pair_rows(records, lookback=0)
        rows_one = sadp.build_adjacent_common_pair_rows(records, lookback=1)
        self.assertEqual(rows_zero, rows_one)

    def test_write_csv_and_plot(self) -> None:
        """Should write CSV rows and produce histogram image."""
        rows = sadp.build_adjacent_common_pair_rows(sadp.build_event_records(make_data()))
        count_rows = sadp.build_shared_pair_count_rows(
            sadp.build_event_records(make_data())
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "out.csv"
            png_path = root / "out.png"
            count_csv_path = root / "count_out.csv"
            count_png_path = root / "count_out.png"
            count_scatter_path = root / "count_scatter.png"

            sadp.write_adjacent_pair_csv(csv_path, rows)
            sadp.plot_adjacent_pair_delta_histogram(png_path, rows, bins=16)
            sadp.write_shared_pair_count_csv(count_csv_path, count_rows)
            sadp.plot_shared_pair_count_histogram(count_png_path, count_rows, bins=8)
            sadp.plot_shared_pair_count_vs_time(count_scatter_path, count_rows)
            du_count_rows = sadp.build_shared_du_count_rows(
                sadp.build_event_records(make_data())
            )
            du_count_png_path = root / "du_count_out.png"
            du_count_scatter_path = root / "du_count_scatter.png"
            sadp.plot_shared_du_count_histogram(
                du_count_png_path,
                du_count_rows,
                bins=8,
            )
            sadp.plot_shared_du_count_vs_time(
                du_count_scatter_path,
                du_count_rows,
            )

            with csv_path.open("r", encoding="utf-8") as file_obj:
                csv_rows = list(csv.DictReader(file_obj))

            self.assertEqual(len(csv_rows), 3)
            self.assertEqual(csv_rows[0]["du_a"], "1")
            self.assertEqual(csv_rows[0]["curr_event_datetime"], "2026-02-14T12:00:01")
            self.assertTrue(png_path.exists())

            loaded_rows = sadp.read_adjacent_pair_csv(csv_path)
            self.assertEqual(len(loaded_rows), 3)
            self.assertEqual(
                loaded_rows[0]["curr_event_datetime"],
                "2026-02-14T12:00:01",
            )

            loaded_count_rows = sadp.read_shared_pair_count_csv(count_csv_path)
            self.assertEqual(len(loaded_count_rows), 3)
            self.assertEqual(loaded_count_rows[0]["shared_pair_count"], 1)
            self.assertTrue(count_png_path.exists())
            self.assertTrue(count_scatter_path.exists())
            self.assertTrue(du_count_png_path.exists())
            self.assertTrue(du_count_scatter_path.exists())

    def test_read_adjacent_pair_csv_skips_invalid_rows(self) -> None:
        """Cache reader should skip bad rows and keep valid ones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "bad_rows.csv"
            csv_path.write_text(
                "prev_event_number,curr_event_number,prev_gps_time,curr_gps_time,du_a,du_b,prev_pair_delta_ns,curr_pair_delta_ns,adjacent_pair_delta_ns,abs_adjacent_pair_delta_ns,curr_event_datetime\n"
                "100,101,10,11,1,2,40.0,40.0,0.0,0.0,2026-02-14T12:00:01\n"
                "100,102,10,12,1,3,40.0,bad,0.0,0.0,2026-02-14T12:00:02\n",
                encoding="utf-8",
            )

            rows = sadp.read_adjacent_pair_csv(csv_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["prev_event_number"], 100)

    def test_plot_skip_on_empty_rows(self) -> None:
        """Plot function should skip file creation when rows are empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "empty.png"
            sadp.plot_adjacent_pair_delta_histogram(png_path, [], bins=10)
            self.assertFalse(png_path.exists())

            scatter_path = Path(tmpdir) / "empty_scatter.png"
            sadp.plot_adjacent_pair_delta_vs_time(scatter_path, [])
            self.assertFalse(scatter_path.exists())

            count_hist_path = Path(tmpdir) / "empty_count.png"
            sadp.plot_shared_pair_count_histogram(count_hist_path, [], bins=10)
            self.assertFalse(count_hist_path.exists())

            count_scatter_path = Path(tmpdir) / "empty_count_scatter.png"
            sadp.plot_shared_pair_count_vs_time(count_scatter_path, [])
            self.assertFalse(count_scatter_path.exists())

            du_count_hist_path = Path(tmpdir) / "empty_du_count.png"
            sadp.plot_shared_du_count_histogram(du_count_hist_path, [], bins=10)
            self.assertFalse(du_count_hist_path.exists())

            du_count_scatter_path = Path(tmpdir) / "empty_du_count_scatter.png"
            sadp.plot_shared_du_count_vs_time(du_count_scatter_path, [])
            self.assertFalse(du_count_scatter_path.exists())

    def test_parse_args(self) -> None:
        """CLI parser should parse yaml file and optional flags."""
        with patch(
            "sys.argv",
            ["stats_lookback.py", "in.yaml", "--bins", "33", "--lookback", "4"],
        ):
            args = sadp.parse_args()
        self.assertEqual(args.yaml_file, "in.yaml")
        self.assertEqual(args.bins, 33)
        self.assertEqual(args.lookback, 4)
        self.assertFalse(args.no_plot)
        self.assertFalse(args.force_recompute)
        self.assertEqual(args.offset_file, "2025-10-28_beacon_25Hz_offset.txt")

    def test_parse_args_force_recompute(self) -> None:
        """CLI parser should parse force-recompute flag."""
        with patch("sys.argv", ["stats_lookback.py", "in.yaml", "--force-recompute"]):
            args = sadp.parse_args()
        self.assertTrue(args.force_recompute)

    def test_parse_args_default_lookback(self) -> None:
        """CLI parser should use lookback=10 by default."""
        with patch("sys.argv", ["stats_lookback.py", "in.yaml"]):
            args = sadp.parse_args()
        self.assertEqual(args.lookback, 10)

    def test_parse_args_datetime_range(self) -> None:
        """CLI parser should parse datetime range options."""
        with patch(
            "sys.argv",
            [
                "stats_lookback.py",
                "in.yaml",
                "--start-datetime",
                "2026-02-14T12:00:01",
                "--end-datetime",
                "2026-02-14T12:00:02",
            ],
        ):
            args = sadp.parse_args()

        self.assertEqual(args.start_datetime, datetime(2026, 2, 14, 12, 0, 1))
        self.assertEqual(args.end_datetime, datetime(2026, 2, 14, 12, 0, 2))

    def test_named_default_dict_and_datetime_helper_branches(self) -> None:
        """Cover NamedDefaultDict strict access and datetime helper branches."""
        row = sadp.make_named_row(("a", "b"), a=1, b=2)

        with self.assertRaises(TypeError):
            row[0] = 5

        row["b"] = 9
        self.assertEqual(row["a"], 1)
        self.assertEqual(row, {"a": 1, "b": 9})

        with self.assertRaises(TypeError):
            _ = row[0]

        self.assertEqual(sadp.parse_event_datetime({"datetime": "bad"}), "")

        with self.assertRaises(Exception):
            sadp.parse_cli_datetime("2026/02/14 12:00:01")

        self.assertIsNone(sadp.parse_payload_datetime({"datetime": ""}))
        self.assertIsNone(sadp.parse_payload_datetime({"datetime": "bad"}))

        event_dt = datetime(2026, 2, 14, 12, 0, 3)
        self.assertFalse(
            sadp.in_datetime_range(
                event_dt,
                datetime(2026, 2, 14, 12, 0, 1),
                datetime(2026, 2, 14, 12, 0, 2),
            )
        )

    def test_filter_offsets_and_apply_ticks_uncovered_branches(self) -> None:
        """Cover invalid-datetime filtering, offset parsing, and tick branches."""
        filtered = sadp.filter_data_by_datetime_range(
            {
                "good": {"datetime": "2026-02-14T12:00:01"},
                "bad": {"datetime": "bad"},
            },
            datetime(2026, 2, 14, 12, 0, 1),
            datetime(2026, 2, 14, 12, 0, 1),
        )
        self.assertEqual(list(filtered.keys()), ["good"])

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.assertEqual(sadp.load_du_time_offsets(root / "missing.txt"), {})

            offset_file = root / "offset.txt"
            offset_file.write_text(
                "# comment\n"
                "\n"
                "bad_line\n"
                "101, abc\n"
                "102, 1.5\n",
                encoding="utf-8",
            )
            offsets = sadp.load_du_time_offsets(offset_file)
            self.assertEqual(offsets, {"102": 1.5})

        fig, ax = plt.subplots()
        try:
            sadp.apply_time_ticks(ax, [], [])
            sadp.apply_time_ticks(ax, [1], ["T1"], max_ticks=0)
            sadp.apply_time_ticks(ax, [1], ["T1"], max_ticks=1)
            sadp.apply_time_ticks(ax, [1, 2], ["", ""], max_ticks=2)
        finally:
            plt.close(fig)

    def test_main_invalid_datetime_range(self) -> None:
        """Main should fail when start datetime is later than end datetime."""
        with patch(
            "sys.argv",
            [
                "stats_lookback.py",
                "in.yaml",
                "--start-datetime",
                "2026-02-14T12:00:03",
                "--end-datetime",
                "2026-02-14T12:00:02",
            ],
        ):
            self.assertEqual(sadp.main(), 2)

    def test_main_missing_file(self) -> None:
        """Main should return 2 when input file does not exist."""
        with patch("sys.argv", ["stats_lookback.py", "definitely_missing.yaml"]):
            code = sadp.main()
        self.assertEqual(code, 2)

    def test_main_yaml_read_error(self) -> None:
        """Main should return 2 when YAML structure is invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "bad.yaml"
            yaml_path.write_text(yaml.safe_dump([1, 2]), encoding="utf-8")
            with patch("sys.argv", ["stats_lookback.py", str(yaml_path)]):
                code = sadp.main()
            self.assertEqual(code, 2)

    def test_main_no_plot(self) -> None:
        """Main should write a single CSV and return 0 with --no-plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "sample.yaml"
            yaml_path.write_text(yaml.safe_dump(make_data()), encoding="utf-8")

            with patch(
                "sys.argv",
                [
                    "stats_lookback.py",
                    str(yaml_path),
                    "--no-plot",
                ],
            ):
                code = sadp.main()

            self.assertEqual(code, 0)
            out_csv = root / "sample_lookback10_common_du_pair_delta_distribution.csv"
            self.assertTrue(out_csv.exists())
            self.assertFalse(
                (
                    root
                    / "sample_lookback10_shared_du_pair_count_distribution.csv"
                ).exists()
            )

    def test_main_no_plot_with_datetime_range(self) -> None:
        """Main should filter events by datetime range before lookback compare."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "sample_range.yaml"
            yaml_path.write_text(yaml.safe_dump(make_data()), encoding="utf-8")

            with patch(
                "sys.argv",
                [
                    "stats_lookback.py",
                    str(yaml_path),
                    "--no-plot",
                    "--start-datetime",
                    "2026-02-14T12:00:01",
                    "--end-datetime",
                    "2026-02-14T12:00:02",
                ],
            ):
                code = sadp.main()

            self.assertEqual(code, 0)
            out_csv = (
                root
                / "sample_range_lookback10_common_du_pair_delta_distribution.csv"
            )
            self.assertTrue(out_csv.exists())
            with out_csv.open("r", encoding="utf-8") as file_obj:
                rows = list(csv.DictReader(file_obj))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["prev_event_number"], "101")
            self.assertEqual(rows[0]["curr_event_number"], "102")

    def test_main_with_plot_and_empty_rows(self) -> None:
        """Main should still return 0 when plotting with no adjacent samples."""
        one_event_data = {
            "1": {
                "event_number": 1,
                "gps_time": 10,
                "index": 0,
                "time": {"1": 100.0, "2": 110.0},
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "single.yaml"
            yaml_path.write_text(yaml.safe_dump(one_event_data), encoding="utf-8")

            with patch("sys.argv", ["stats_lookback.py", str(yaml_path)]):
                code = sadp.main()

            self.assertEqual(code, 0)
            out_csv = root / "single_lookback10_common_du_pair_delta_distribution.csv"
            out_png = root / "single_lookback10_common_du_pair_delta_hist.png"
            self.assertTrue(out_csv.exists())
            self.assertFalse(out_png.exists())

    def test_main_with_plot_and_rows(self) -> None:
        """Main should generate PNG and hit plot-written log branch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "sample_plot.yaml"
            yaml_path.write_text(yaml.safe_dump(make_data()), encoding="utf-8")

            with patch("sys.argv", ["stats_lookback.py", str(yaml_path)]):
                code = sadp.main()

            self.assertEqual(code, 0)
            out_png = root / "sample_plot_lookback10_common_du_pair_delta_hist.png"
            out_scatter = root / "sample_plot_lookback10_common_du_pair_delta_vs_time.png"
            count_hist = root / "sample_plot_lookback10_shared_du_pair_count_hist.png"
            count_scatter = (
                root / "sample_plot_lookback10_shared_du_pair_count_vs_time.png"
            )
            du_count_hist = (
                root / "sample_plot_lookback10_shared_du_count_hist.png"
            )
            du_count_scatter = (
                root / "sample_plot_lookback10_shared_du_count_vs_time.png"
            )
            self.assertTrue(out_png.exists())
            self.assertTrue(out_scatter.exists())
            self.assertTrue(count_hist.exists())
            self.assertTrue(count_scatter.exists())
            self.assertTrue(du_count_hist.exists())
            self.assertTrue(du_count_scatter.exists())

    def test_main_uses_csv_cache_without_yaml(self) -> None:
        """Main should read existing CSV cache directly even if YAML is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "cache_source.yaml"
            csv_path = root / "cache_source_lookback10_common_du_pair_delta_distribution.csv"

            cached_rows = [
                (100, 101, 10, 11, "1", "2", 40.0, 40.0, 0.0, 0.0, "2026-02-14T12:00:01")
            ]
            sadp.write_adjacent_pair_csv(csv_path, cached_rows)

            with patch("sys.argv", ["stats_lookback.py", str(yaml_path), "--no-plot"]):
                code = sadp.main()

            self.assertEqual(code, 0)

    def test_main_cache_missing_meta_falls_back_recompute(self) -> None:
        """When cache meta is missing, main should fall back to YAML recompute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "cache_recompute.yaml"
            yaml_path.write_text(yaml.safe_dump(make_data()), encoding="utf-8")

            csv_path = (
                root
                / "cache_recompute_lookback10_common_du_pair_delta_distribution.csv"
            )
            sadp.write_adjacent_pair_csv(
                csv_path,
                [
                    (
                        100,
                        101,
                        10,
                        11,
                        "1",
                        "2",
                        40.0,
                        40.0,
                        999.0,
                        999.0,
                        "2026-02-14T12:00:01",
                    )
                ],
            )

            with patch("sys.argv", ["stats_lookback.py", str(yaml_path), "--no-plot"]):
                code = sadp.main()

            self.assertEqual(code, 0)
            with csv_path.open("r", encoding="utf-8") as file_obj:
                rows = list(csv.DictReader(file_obj))
            self.assertEqual(len(rows), 3)

    def test_main_offset_drift_falls_back_recompute(self) -> None:
        """Offset file change should invalidate cache and trigger recompute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "offset_drift.yaml"
            yaml_path.write_text(yaml.safe_dump(make_data()), encoding="utf-8")

            offset_path = root / "offset.txt"
            offset_path.write_text("1,0\n2,0\n3,0\n4,0\n", encoding="utf-8")

            with patch(
                "sys.argv",
                [
                    "stats_lookback.py",
                    str(yaml_path),
                    "--no-plot",
                    "--offset-file",
                    str(offset_path),
                ],
            ):
                first_code = sadp.main()
            self.assertEqual(first_code, 0)

            csv_path = root / "offset_drift_lookback10_common_du_pair_delta_distribution.csv"
            sadp.write_adjacent_pair_csv(
                csv_path,
                [
                    (
                        100,
                        101,
                        10,
                        11,
                        "1",
                        "2",
                        40.0,
                        40.0,
                        999.0,
                        999.0,
                        "2026-02-14T12:00:01",
                    )
                ],
            )

            offset_path.write_text("1,10\n2,0\n3,0\n4,0\n", encoding="utf-8")

            with patch(
                "sys.argv",
                [
                    "stats_lookback.py",
                    str(yaml_path),
                    "--no-plot",
                    "--offset-file",
                    str(offset_path),
                ],
            ):
                second_code = sadp.main()

            self.assertEqual(second_code, 0)
            with csv_path.open("r", encoding="utf-8") as file_obj:
                rows = list(csv.DictReader(file_obj))
            self.assertEqual(len(rows), 3)

    def test_main_cache_plot_with_yaml_success_and_warning_paths(self) -> None:
        """Cover cache plot path with YAML success and warning branches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "cache_plot.yaml"
            csv_path = root / "cache_plot_lookback10_common_du_pair_delta_distribution.csv"

            cached_rows = [
                (100, 101, 10, 11, "1", "2", 40.0, 40.0, 0.0, 0.0, "2026-02-14T12:00:01")
            ]
            sadp.write_adjacent_pair_csv(csv_path, cached_rows)
            yaml_path.write_text(yaml.safe_dump(make_data()), encoding="utf-8")

            with patch("sys.argv", ["stats_lookback.py", str(yaml_path)]):
                code = sadp.main()
            self.assertEqual(code, 0)
            self.assertTrue(
                (root / "cache_plot_lookback10_common_du_pair_delta_hist.png").exists()
            )
            self.assertTrue(
                (root / "cache_plot_lookback10_shared_du_count_hist.png").exists()
            )

            with patch(
                "stats.stats_lookback.read_yaml_events",
                side_effect=RuntimeError("boom"),
            ), patch("sys.argv", ["stats_lookback.py", str(yaml_path)]):
                code = sadp.main()
            self.assertEqual(code, 0)

    def test_main_cache_bad_structure_falls_back_recompute(self) -> None:
        """Malformed cache columns should trigger recompute when YAML exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "cache_bad_structure.yaml"
            yaml_path.write_text(yaml.safe_dump(make_data()), encoding="utf-8")
            csv_path = (
                root
                / "cache_bad_structure_lookback10_common_du_pair_delta_distribution.csv"
            )
            meta_path = sadp.cache_meta_path(csv_path)

            csv_path.write_text(
                "prev_event_number,curr_event_number,prev_gps_time,curr_gps_time,du_a,du_b\n"
                "100,101,10,11,1,2\n",
                encoding="utf-8",
            )

            sadp.write_cache_meta(
                meta_path,
                sadp.build_cache_meta(
                    yaml_path,
                    Path("2025-10-28_beacon_25Hz_offset.txt"),
                    10,
                    None,
                    None,
                ),
            )

            with patch("sys.argv", ["stats_lookback.py", str(yaml_path), "--no-plot"]):
                code = sadp.main()

            self.assertEqual(code, 0)
            with csv_path.open("r", encoding="utf-8") as file_obj:
                rows = list(csv.DictReader(file_obj))
            self.assertEqual(len(rows), 3)

    def test_main_force_recompute_ignores_cache(self) -> None:
        """Force recompute should not use cache and should require YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "force_source.yaml"
            csv_path = root / "force_source_lookback10_common_du_pair_delta_distribution.csv"

            cached_rows = [
                (100, 101, 10, 11, "1", "2", 40.0, 40.0, 0.0, 0.0, "2026-02-14T12:00:01")
            ]
            sadp.write_adjacent_pair_csv(csv_path, cached_rows)

            with patch(
                "sys.argv",
                ["stats_lookback.py", str(yaml_path), "--no-plot", "--force-recompute"],
            ):
                code = sadp.main()

            self.assertEqual(code, 2)

    def test_module_main_entrypoint(self) -> None:
        """Executing module as __main__ should raise SystemExit with code 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "entry.yaml"
            yaml_path.write_text(yaml.safe_dump(make_data()), encoding="utf-8")

            with patch("sys.argv", ["stats_lookback.py", str(yaml_path), "--no-plot"]):
                with self.assertRaises(SystemExit) as ctx:
                    runpy.run_path(str(Path(sadp.__file__)), run_name="__main__")

            self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()