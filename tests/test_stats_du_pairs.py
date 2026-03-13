"""Unit tests for stats_du_pairs.py."""

from __future__ import annotations

import csv
import runpy
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

import stats.stats_du_pairs as sdn


def make_sample_data() -> dict:
    """Create a minimal Trigger-like payload dict."""
    return {
        "evtA": {
            "event_number": 100,
            "gps_time": 1771027344,
            "datetime": "2026-03-05T12:30:44",
            "time": {
                "101": 1000,
                "102": 1015,
                "103": 980,
            },
        },
        "evtB": {
            "event_number": 101,
            "gps_time": 1771027345,
            "datetime": "2026-03-05T12:30:45",
            "time": {
                "101": 2000,
                "103": 2025,
            },
        },
    }


def make_observed_row(values: tuple) -> sdn.ObservedPairDelta:
    """Build one observed row using stats_du_pairs schema."""
    return sdn.make_named_row(
        sdn.OBSERVED_PAIR_DELTA_FIELDS,
        event_number=values[0],
        event_time=values[1],
        du_a=values[2],
        du_b=values[3],
        observed_delta_ns=values[4],
        observed_abs_delta_ns=values[5],
    )


def make_expected_row(values: tuple) -> sdn.ExpectedPairDelta:
    """Build one expected row using stats_du_pairs schema."""
    return sdn.make_named_row(
        sdn.EXPECTED_PAIR_DELTA_FIELDS,
        du_a=values[0],
        du_b=values[1],
        distance_m=values[2],
        theoretical_delta_ns=values[3],
    )


def make_distribution_row(values: tuple) -> sdn.PairDistributionRow:
    """Build one pair-distribution row using stats_du_pairs schema."""
    return sdn.make_named_row(
        sdn.PAIR_DISTRIBUTION_FIELDS,
        event_number=values[0],
        event_time=values[1],
        du_a=values[2],
        du_b=values[3],
        observed_delta_ns=values[4],
        observed_abs_delta_ns=values[5],
        theoretical_delta_ns=values[6],
        distance_m=values[7],
        event_datetime=values[8],
    )


def make_event_count_row(values: tuple) -> sdn.EventPairCountRow:
    """Build one event pair-count row using stats_du_pairs schema."""
    return sdn.make_named_row(
        sdn.EVENT_PAIR_COUNT_FIELDS,
        event_number=values[0],
        event_time=values[1],
        pair_count=values[2],
        event_datetime=values[3],
    )


def make_event_ratio_row(values: tuple) -> sdn.EventMaxRatioRow:
    """Build one event max-ratio row using stats_du_pairs schema."""
    return sdn.make_named_row(
        sdn.EVENT_MAX_RATIO_FIELDS,
        event_number=values[0],
        event_time=values[1],
        max_observed_theoretical_ratio=values[2],
    )


class TestStatsDuPairs(unittest.TestCase):
    """Tests for time-map and DU-pair delta statistics."""

    def test_parse_args(self) -> None:
        """Should parse CLI arguments correctly."""
        old_argv = sys.argv
        try:
            sys.argv = [
                "stats_du_pairs.py",
                "input.yaml",
                "--det-pos",
                "det.txt",
                "--offset-file",
                "offset.txt",
                "--start-datetime",
                "2026-03-05T12:00:00",
                "--end-datetime",
                "2026-03-05T13:00:00",
                "--no-plot",
            ]
            args = sdn.parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.yaml_file, "input.yaml")
        self.assertEqual(args.det_pos, "det.txt")
        self.assertEqual(args.offset_file, "offset.txt")
        self.assertEqual(args.start_datetime, datetime(2026, 3, 5, 12, 0, 0))
        self.assertEqual(args.end_datetime, datetime(2026, 3, 5, 13, 0, 0))
        self.assertTrue(args.no_plot)

    def test_parse_args_invalid_datetime(self) -> None:
        """Should reject invalid datetime CLI format."""
        old_argv = sys.argv
        try:
            sys.argv = [
                "stats_du_pairs.py",
                "input.yaml",
                "--start-datetime",
                "2026-03-05 12:00:00",
            ]
            with self.assertRaises(SystemExit):
                sdn.parse_args()
        finally:
            sys.argv = old_argv

    def test_parse_args_help(self) -> None:
        """Should print help and exit cleanly with code 0."""
        old_argv = sys.argv
        try:
            sys.argv = ["stats_du_pairs.py", "-h"]
            with self.assertRaises(SystemExit) as context:
                sdn.parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(context.exception.code, 0)

    def test_read_yaml_events_edge_cases(self) -> None:
        """Should handle empty YAML and non-dict YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            empty_yaml = root / "empty.yaml"
            list_yaml = root / "list.yaml"

            empty_yaml.write_text("", encoding="utf-8")
            list_yaml.write_text("- 1\n- 2\n", encoding="utf-8")

            self.assertEqual(sdn.read_yaml_events(empty_yaml), {})
            with self.assertRaises(ValueError):
                sdn.read_yaml_events(list_yaml)

    def test_parse_du_ns_map_edge_cases(self) -> None:
        """Should skip invalid time formats and non-numeric values."""
        self.assertEqual(sdn.parse_du_ns_map({"time": []}), {})

        payload = {"time": {"101": "10", "102": "bad"}}
        parsed = sdn.parse_du_ns_map(payload)
        self.assertEqual(parsed["101"], 10.0)
        self.assertNotIn("102", parsed)

        list_payload = {"time": {"101": [1, "2.5", "bad"], "102": ["x"]}}
        parsed_list = sdn.parse_du_ns_map(list_payload)
        self.assertEqual(parsed_list["101"], 2.5)
        self.assertNotIn("102", parsed_list)

    def test_build_observed_pair_deltas(self) -> None:
        """Should build pairwise observed deltas from each event."""
        rows = sdn.build_observed_pair_deltas(make_sample_data())
        self.assertEqual(len(rows), 4)

        first_evt_rows = [row for row in rows if row["event_number"] == 100]
        self.assertEqual(len(first_evt_rows), 3)

        pair_101_102 = [
            row
            for row in first_evt_rows
            if row["du_a"] == "101" and row["du_b"] == "102"
        ][0]
        self.assertEqual(pair_101_102["event_time"], 1771027344)
        self.assertEqual(pair_101_102["observed_delta_ns"], 15.0)
        self.assertEqual(pair_101_102["observed_abs_delta_ns"], 15.0)

    def test_build_observed_pair_deltas_with_offsets(self) -> None:
        """Should apply DU offsets when computing observed pair deltas."""
        offsets = {"101": 2.0, "102": -3.0}
        rows = sdn.build_observed_pair_deltas(make_sample_data(), offsets)

        first_evt_rows = [row for row in rows if row["event_number"] == 100]
        pair_101_102 = [
            row
            for row in first_evt_rows
            if row["du_a"] == "101" and row["du_b"] == "102"
        ][0]
        self.assertEqual(pair_101_102["observed_delta_ns"], 20.0)
        self.assertEqual(pair_101_102["observed_abs_delta_ns"], 20.0)

    def test_load_du_time_offsets(self) -> None:
        """Should parse offsets file and ignore malformed lines/sigma column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            offset_file = root / "offset.txt"
            offset_file.write_text(
                "# comment line\n"
                "\n"
                "101, -42.93, 10\n"
                "bad_line\n"
                "102, abc, 5\n"
                "103, 1.5\n",
                encoding="utf-8",
            )

            offsets = sdn.load_du_time_offsets(offset_file)
            self.assertEqual(offsets["101"], -42.93)
            self.assertEqual(offsets["103"], 1.5)
            self.assertNotIn("102", offsets)

    def test_load_du_time_offsets_missing_file(self) -> None:
        """Should return empty map when offset file does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            offsets = sdn.load_du_time_offsets(root / "not_exists.txt")
            self.assertEqual(offsets, {})

    def test_datetime_parsing_and_formatting_branches(self) -> None:
        """Should cover datetime parsing edge branches and formatting paths."""
        date_str, time_str = sdn.parse_event_date_time(
            {"datetime": "2026-03-05T12:30:45"}
        )
        self.assertEqual(date_str, "20260305")
        self.assertEqual(time_str, "123045")

        invalid_date, invalid_time = sdn.parse_event_date_time({"datetime": "bad"})
        self.assertEqual(invalid_date, "")
        self.assertEqual(invalid_time, "")
        self.assertEqual(sdn.parse_event_date_time({"datetime": ""}), ("", ""))

        formatted = sdn.format_event_datetime("20260305", "123045")
        self.assertEqual(formatted, "2026-03-05T12:30:45")

        fallback_text = sdn.format_event_datetime("20260305", "bad")
        self.assertEqual(fallback_text, "20260305 bad")
        self.assertEqual(sdn.format_event_datetime("", "123045"), "")

    def test_datetime_helpers_additional_uncovered_branches(self) -> None:
        """Should cover remaining datetime helper branch conditions."""
        self.assertIsNone(sdn.parse_event_datetime({"datetime": ""}))
        self.assertIsNone(sdn.parse_row_datetime(""))

        event_dt = datetime(2026, 3, 5, 13, 0, 1)
        start_dt = datetime(2026, 3, 5, 12, 0, 0)
        end_dt = datetime(2026, 3, 5, 13, 0, 0)
        self.assertFalse(sdn.in_datetime_range(event_dt, start_dt, end_dt))

    def test_event_time_label_map_branch_coverage(self) -> None:
        """Should cover invalid/duplicate gps_time branches in label map builder."""
        data = {
            "evt_a": {"gps_time": "bad", "datetime": "2026-03-05T12:30:45"},
            "evt_b": {"gps_time": -1, "datetime": "2026-03-05T12:30:46"},
            "evt_c": {"gps_time": 100, "datetime": "2026-03-05T12:30:47"},
            "evt_d": {"gps_time": 100, "datetime": "2026-03-05T12:30:48"},
        }
        label_map = sdn.build_event_time_label_map(data)
        self.assertIn(100, label_map)
        self.assertEqual(len(label_map), 1)

    def test_event_time_label_map_from_rows_and_apply_ticks(self) -> None:
        """Should build labels from cached rows and apply datetime x-ticks."""
        rows = [
            make_distribution_row((100, -1, "101", "102", 1.0, 1.0, 1.0, 1.0, "ignored")),
            make_distribution_row((101, 1771027344, "101", "103", 2.0, 2.0, 1.0, 1.0, "2026-03-05T12:30:44")),
            make_distribution_row((102, 1771027344, "101", "104", 3.0, 3.0, 1.0, 1.0, "dup")),
            make_distribution_row((103, 1771027345, "101", "105", 4.0, 4.0, 1.0, 1.0, "")),
        ]
        label_map = sdn.build_event_time_label_map_from_rows(rows)
        self.assertEqual(label_map[1771027344], "2026-03-05T12:30:44")

        fig, ax = sdn.plt.subplots(figsize=(4, 3))
        try:
            sdn.apply_time_ticks(ax, np.array([]), label_map)
            sdn.apply_time_ticks(
                ax,
                np.array([1771027344, 1771027345]),
                label_map,
            )
            tick_labels = [label.get_text() for label in ax.get_xticklabels()]
            self.assertTrue(any("2026-03-05T" in text for text in tick_labels))

            sdn.apply_time_ticks(ax, [1, 2, 3], label_map, max_ticks=0)
        finally:
            sdn.plt.close(fig)

    def test_filter_events_by_datetime_range(self) -> None:
        """Should filter YAML events by inclusive datetime range."""
        data = {
            "evt_a": {
                "event_number": 1,
                "datetime": "2026-03-05T12:00:00",
                "time": {"101": 1},
            },
            "evt_b": {
                "event_number": 2,
                "datetime": "2026-03-05T13:00:00",
                "time": {"101": 2},
            },
            "evt_c": {
                "event_number": 3,
                "datetime": "bad",
                "time": {"101": 3},
            },
        }

        start_dt = datetime(2026, 3, 5, 12, 30, 0)
        end_dt = datetime(2026, 3, 5, 13, 0, 0)
        filtered = sdn.filter_events_by_datetime_range(data, start_dt, end_dt)
        self.assertEqual(list(filtered.keys()), ["evt_b"])

    def test_filter_distribution_rows_by_datetime_range(self) -> None:
        """Should filter cached distribution rows by datetime text field."""
        rows = [
            make_distribution_row((100, 1, "101", "102", 1.0, 1.0, 1.0, 1.0, "2026-03-05T12:00:00")),
            make_distribution_row((101, 2, "101", "103", 1.0, 1.0, 1.0, 1.0, "2026-03-05T13:00:00")),
            make_distribution_row((102, 3, "101", "104", 1.0, 1.0, 1.0, 1.0, "bad")),
        ]

        start_dt = datetime(2026, 3, 5, 12, 30, 0)
        end_dt = datetime(2026, 3, 5, 13, 0, 0)
        filtered = sdn.filter_distribution_rows_by_datetime_range(
            rows,
            start_dt,
            end_dt,
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["event_number"], 101)

    def test_non_zero_floor_magnitude_fallback(self) -> None:
        """Should return fallback floor when no positive values are present."""
        self.assertEqual(sdn.non_zero_floor_magnitude([0.0, -1.0]), 1e-12)

    def test_build_expected_pair_deltas(self) -> None:
        """Should compute expected pair delta from geometry distance/c."""
        detector_positions = {
            "101": np.array([0.0, 0.0, 0.0]),
            "102": np.array([299_792_458.0, 0.0, 0.0]),
            "103": np.array([0.0, 299_792_458.0, 0.0]),
        }

        rows = sdn.build_expected_pair_deltas(detector_positions, ["101", "102"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["du_a"], "101")
        self.assertEqual(rows[0]["du_b"], "102")
        self.assertAlmostEqual(rows[0]["theoretical_delta_ns"], 1e9, places=3)

        rows_all = sdn.build_expected_pair_deltas(detector_positions)
        self.assertEqual(len(rows_all), 3)

    def test_theoretical_cache_and_distribution_rows(self) -> None:
        """Should persist/reload theoretical cache and join distribution rows."""
        observed_rows = [
            make_observed_row((100, 1771027344, "101", "102", 15.0, 15.0)),
            make_observed_row((101, 1771027345, "101", "102", 10.0, 10.0)),
        ]
        expected_rows = [make_expected_row(("101", "102", 10.0, 33.0))]

        rows = sdn.build_pair_distribution_rows(observed_rows, expected_rows)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["event_number"], 100)
        self.assertEqual(rows[0]["event_time"], 1771027344)
        self.assertEqual(rows[0]["theoretical_delta_ns"], 33.0)
        self.assertEqual(rows[0]["distance_m"], 10.0)

        rows_no_match = sdn.build_pair_distribution_rows(
            observed_rows,
            [make_expected_row(("103", "104", 1.0, 2.0))],
        )
        self.assertEqual(rows_no_match, [])

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            det_pos_path = root / "det.txt"
            det_pos_path.write_text("101 0 0 0\n", encoding="utf-8")

            cache_path = sdn.theoretical_cache_path(det_pos_path)
            sdn.write_theoretical_cache(cache_path, expected_rows)
            loaded = sdn.read_theoretical_cache(cache_path)
            self.assertEqual(loaded, expected_rows)

    def test_read_cache_csv_skips_invalid_rows(self) -> None:
        """Cache readers should skip invalid rows instead of crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            theoretical_csv = root / "det_du_pair_theoretical.csv"
            with theoretical_csv.open("w", newline="", encoding="utf-8") as file_obj:
                writer = csv.writer(file_obj)
                writer.writerow(["du_a", "du_b", "distance_m", "theoretical_delta_ns"])
                writer.writerow(["101", "102", "3.0", "10.0"])
                writer.writerow(["101", "103", "4.0", "bad"])

            loaded_theoretical = sdn.read_theoretical_cache(theoretical_csv)
            self.assertEqual(len(loaded_theoretical), 1)
            self.assertEqual(loaded_theoretical[0]["du_a"], "101")
            self.assertEqual(loaded_theoretical[0]["du_b"], "102")

            distribution_csv = root / "Trigger_cached_du_pair_delta_distribution.csv"
            with distribution_csv.open("w", newline="", encoding="utf-8") as file_obj:
                writer = csv.writer(file_obj)
                writer.writerow(
                    [
                        "event_number",
                        "event_time",
                        "du_a",
                        "du_b",
                        "observed_delta_ns",
                        "observed_abs_delta_ns",
                        "theoretical_delta_ns",
                        "distance_m",
                        "event_datetime",
                    ]
                )
                writer.writerow([100, 1771027344, "101", "102", 10.0, 10.0, 8.0, 3.0, "2026-03-05T12:30:44"])
                writer.writerow([101, "bad-time", "101", "103", 11.0, 11.0, 9.0, 4.0, "2026-03-05T12:30:45"])
                writer.writerow([102, 1771027346, "101", "104", 12.0, 12.0, "bad-theory", 5.0, "2026-03-05T12:30:46"])

            loaded_distribution = sdn.read_pair_distribution_csv(distribution_csv)
            self.assertEqual(len(loaded_distribution), 1)
            self.assertEqual(loaded_distribution[0]["event_time"], 1771027344)
            self.assertEqual(loaded_distribution[0]["theoretical_delta_ns"], 8.0)

    def test_main_fallback_recompute_on_invalid_distribution_cache_schema(self) -> None:
        """Main should fallback to recompute when cache schema is invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "Trigger_invalid_cache.yaml"
            det_path = root / "det.txt"
            distribution_csv = root / "Trigger_invalid_cache_du_pair_delta_distribution.csv"

            yaml_path.write_text(
                yaml.safe_dump(make_sample_data(), sort_keys=False),
                encoding="utf-8",
            )
            det_path.write_text(
                "101 0 0 0\n102 3 0 0\n103 0 4 0\n",
                encoding="utf-8",
            )

            distribution_csv.write_text("bad,columns\n1,2\n", encoding="utf-8")

            old_parse_args = sdn.parse_args
            try:
                sdn.parse_args = lambda: type(
                    "Args",
                    (),
                    {
                        "yaml_file": str(yaml_path),
                        "det_pos": str(det_path),
                        "offset_file": str(root / "offset.txt"),
                        "no_plot": True,
                        "force_recompute": False,
                        "start_datetime": None,
                        "end_datetime": None,
                    },
                )()
                exit_code = sdn.main()
            finally:
                sdn.parse_args = old_parse_args

            self.assertEqual(exit_code, 0)
            with distribution_csv.open("r", encoding="utf-8") as file_obj:
                first_line = file_obj.readline().strip()
            self.assertIn("event_number", first_line)

    def test_plot_functions(self) -> None:
        """Should execute pair plot for empty and non-empty inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            sdn.plot_pair_delta_distribution(root / "pair_empty.png", [])
            sdn.plot_observed_expected_ratio(root / "ratio_empty.png", [])
            sdn.plot_delta_vs_event_time(root / "delta_time_empty.png", [])
            sdn.plot_ratio_vs_event_time(root / "ratio_time_empty.png", [])

            sdn.plot_pair_delta_distribution(
                root / "pair_obs_only.png",
                [1.0, 2.0, 3.0],
            )
            rows = [
                make_distribution_row((100, 1771027344, "101", "102", 9.0, 9.0, 10.0, 3.0, "")),
                make_distribution_row((100, 1771027345, "101", "103", 18.0, 18.0, 20.0, 6.0, "")),
            ]
            sdn.plot_observed_expected_ratio(root / "ratio.png", rows)
            sdn.plot_delta_vs_event_time(root / "delta_time.png", rows)
            sdn.plot_ratio_vs_event_time(root / "ratio_time.png", rows)

            self.assertTrue((root / "pair_obs_only.png").exists())
            self.assertTrue((root / "ratio.png").exists())
            self.assertTrue((root / "delta_time.png").exists())
            self.assertTrue((root / "ratio_time.png").exists())

    def test_build_event_max_ratio_rows(self) -> None:
        """Should aggregate per-event maximum observed/theoretical ratio."""
        rows = [
            make_distribution_row((100, 1771027344, "101", "102", 10.0, 10.0, 5.0, 1.0, "")),
            make_distribution_row((100, 1771027344, "101", "103", 12.0, 12.0, 4.0, 1.0, "")),
            make_distribution_row((101, 1771027345, "101", "104", 8.0, 8.0, 8.0, 1.0, "")),
            make_distribution_row((102, -1, "101", "105", 9.0, 9.0, 3.0, 1.0, "")),
            make_distribution_row((103, 1771027346, "101", "106", 9.0, 9.0, 0.0, 1.0, "")),
        ]

        max_ratio_rows = sdn.build_event_max_ratio_rows(rows)
        self.assertEqual(len(max_ratio_rows), 2)
        self.assertEqual(max_ratio_rows[0]["event_number"], 100)
        self.assertEqual(max_ratio_rows[0]["event_time"], 1771027344)
        self.assertEqual(max_ratio_rows[0]["max_observed_theoretical_ratio"], 3.0)
        self.assertEqual(max_ratio_rows[1]["event_number"], 101)
        self.assertEqual(max_ratio_rows[1]["event_time"], 1771027345)
        self.assertEqual(max_ratio_rows[1]["max_observed_theoretical_ratio"], 1.0)

    def test_build_event_pair_count_rows(self) -> None:
        """Should aggregate per-event DU-pair counts from pair rows."""
        rows = [
            make_distribution_row((100, 1771027344, "101", "102", 10.0, 10.0, 5.0, 1.0, "T1")),
            make_distribution_row((100, 1771027344, "101", "103", 12.0, 12.0, 4.0, 1.0, "")),
            make_distribution_row((101, 1771027345, "101", "104", 8.0, 8.0, 8.0, 1.0, "T2")),
        ]

        count_rows = sdn.build_event_pair_count_rows(rows)
        self.assertEqual(len(count_rows), 2)
        self.assertEqual(count_rows[0]["event_number"], 100)
        self.assertEqual(count_rows[0]["event_time"], 1771027344)
        self.assertEqual(count_rows[0]["pair_count"], 2)
        self.assertEqual(count_rows[0]["event_datetime"], "T1")
        self.assertEqual(count_rows[1]["event_number"], 101)
        self.assertEqual(count_rows[1]["event_time"], 1771027345)
        self.assertEqual(count_rows[1]["pair_count"], 1)
        self.assertEqual(count_rows[1]["event_datetime"], "T2")

    def test_plot_event_pair_count_functions(self) -> None:
        """Should cover per-event pair-count plot empty and non-empty paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            sdn.plot_event_pair_count_vs_time(root / "pair_count_time_empty.png", [])
            sdn.plot_event_pair_count_histogram(root / "pair_count_hist_empty.png", [])
            self.assertFalse((root / "pair_count_time_empty.png").exists())
            self.assertFalse((root / "pair_count_hist_empty.png").exists())

            rows = [
                make_event_count_row((100, 1771027344, 2, "2026-03-05T12:30:44")),
                make_event_count_row((101, 1771027345, 1, "2026-03-05T12:30:45")),
            ]
            label_map = {
                1771027344: "2026-03-05T12:30:44",
                1771027345: "2026-03-05T12:30:45",
            }
            sdn.plot_event_pair_count_vs_time(
                root / "pair_count_time.png",
                rows,
                label_map,
            )
            sdn.plot_event_pair_count_histogram(
                root / "pair_count_hist.png",
                rows,
            )
            self.assertTrue((root / "pair_count_time.png").exists())
            self.assertTrue((root / "pair_count_hist.png").exists())

    def test_plot_event_max_ratio_functions(self) -> None:
        """Should cover per-event max-ratio plot empty and non-empty paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            sdn.plot_event_max_ratio_vs_time(root / "max_ratio_time_empty.png", [])
            sdn.plot_event_max_ratio_histogram(root / "max_ratio_hist_empty.png", [])
            self.assertFalse((root / "max_ratio_time_empty.png").exists())
            self.assertFalse((root / "max_ratio_hist_empty.png").exists())

            rows = [
                make_event_ratio_row((100, 1771027344, 2.0)),
                make_event_ratio_row((101, 1771027345, 4.0)),
            ]
            label_map = {
                1771027344: "2026-03-05T12:30:44",
                1771027345: "2026-03-05T12:30:45",
            }
            sdn.plot_event_max_ratio_vs_time(
                root / "max_ratio_time.png",
                rows,
                label_map,
            )
            sdn.plot_event_max_ratio_histogram(root / "max_ratio_hist.png", rows)
            self.assertTrue((root / "max_ratio_time.png").exists())
            self.assertTrue((root / "max_ratio_hist.png").exists())

    def test_plot_theoretical_delta_functions(self) -> None:
        """Should cover theoretical-delta plot empty and non-empty paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            sdn.plot_theoretical_delta_vs_event_time(
                root / "theoretical_time_empty.png",
                [],
            )
            sdn.plot_theoretical_delta_histogram(
                root / "theoretical_hist_empty.png",
                [],
            )
            self.assertFalse((root / "theoretical_time_empty.png").exists())
            self.assertFalse((root / "theoretical_hist_empty.png").exists())

            invalid_time_rows = [
                make_distribution_row((100, -1, "101", "102", 9.0, 9.0, 10.0, 3.0, "")),
            ]
            sdn.plot_theoretical_delta_vs_event_time(
                root / "theoretical_time_skip.png",
                invalid_time_rows,
            )
            self.assertFalse((root / "theoretical_time_skip.png").exists())

            invalid_hist_rows = [
                make_distribution_row((101, 1771027344, "101", "103", 9.0, 9.0, 0.0, 3.0, "")),
            ]
            sdn.plot_theoretical_delta_histogram(
                root / "theoretical_hist_skip.png",
                invalid_hist_rows,
            )
            self.assertFalse((root / "theoretical_hist_skip.png").exists())

            rows = [
                make_distribution_row((100, 1771027344, "101", "102", 9.0, 9.0, 10.0, 3.0, "")),
                make_distribution_row((101, 1771027345, "101", "103", 18.0, 18.0, 20.0, 6.0, "")),
            ]
            label_map = {
                1771027344: "2026-03-05T12:30:44",
                1771027345: "2026-03-05T12:30:45",
            }
            sdn.plot_theoretical_delta_vs_event_time(
                root / "theoretical_time.png",
                rows,
                label_map,
            )
            sdn.plot_theoretical_delta_histogram(
                root / "theoretical_hist.png",
                rows,
            )
            self.assertTrue((root / "theoretical_time.png").exists())
            self.assertTrue((root / "theoretical_hist.png").exists())

    def test_plot_observed_expected_ratio_zero_theoretical(self) -> None:
        """Ratio plot should skip when all theoretical values are non-positive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = [
                make_distribution_row((100, 1771027344, "101", "102", 9.0, 9.0, 0.0, 3.0, "")),
            ]
            sdn.plot_observed_expected_ratio(root / "ratio_zero.png", rows)
            self.assertFalse((root / "ratio_zero.png").exists())

    def test_time_scatter_skip_paths(self) -> None:
        """Time-based scatter plots should skip when event_time is invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = [
                make_distribution_row((100, -1, "101", "102", 9.0, 9.0, 10.0, 3.0, "")),
            ]
            sdn.plot_delta_vs_event_time(root / "delta_time_skip.png", rows)
            sdn.plot_ratio_vs_event_time(root / "ratio_time_skip.png", rows)
            self.assertFalse((root / "delta_time_skip.png").exists())
            self.assertFalse((root / "ratio_time_skip.png").exists())

    def test_main_error_paths(self) -> None:
        """Main should return error code for missing file and invalid YAML."""
        old_parse_args = sdn.parse_args
        try:
            sdn.parse_args = lambda: type(
                "Args",
                (),
                {
                    "yaml_file": "not_exists.yaml",
                    "det_pos": "_gp65_rtksort.txt",
                    "offset_file": "2025-10-28_beacon_25Hz_offset.txt",
                    "no_plot": True,
                    "force_recompute": False,
                    "start_datetime": None,
                    "end_datetime": None,
                },
            )()
            self.assertEqual(sdn.main(), 2)
        finally:
            sdn.parse_args = old_parse_args

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "bad.yaml"
            det_path = root / "det.txt"
            yaml_path.write_text("- 1\n- 2\n", encoding="utf-8")
            det_path.write_text("101 0 0 0\n", encoding="utf-8")

            old_parse_args = sdn.parse_args
            try:
                sdn.parse_args = lambda: type(
                    "Args",
                    (),
                    {
                        "yaml_file": str(yaml_path),
                        "det_pos": str(det_path),
                        "offset_file": str(root / "offset.txt"),
                        "no_plot": True,
                        "force_recompute": False,
                        "start_datetime": None,
                        "end_datetime": None,
                    },
                )()
                self.assertEqual(sdn.main(), 2)
            finally:
                sdn.parse_args = old_parse_args

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "ok.yaml"
            missing_det_path = root / "det_missing.txt"
            yaml_path.write_text(
                yaml.safe_dump(make_sample_data(), sort_keys=False),
                encoding="utf-8",
            )

            old_parse_args = sdn.parse_args
            try:
                sdn.parse_args = lambda: type(
                    "Args",
                    (),
                    {
                        "yaml_file": str(yaml_path),
                        "det_pos": str(missing_det_path),
                        "offset_file": str(root / "offset.txt"),
                        "no_plot": True,
                        "force_recompute": False,
                        "start_datetime": None,
                        "end_datetime": None,
                    },
                )()
                self.assertEqual(sdn.main(), 2)
            finally:
                sdn.parse_args = old_parse_args

    def test_main_invalid_datetime_range(self) -> None:
        """Main should fail when start datetime is later than end datetime."""
        old_parse_args = sdn.parse_args
        try:
            sdn.parse_args = lambda: type(
                "Args",
                (),
                {
                    "yaml_file": "not_exists.yaml",
                    "det_pos": "_gp65_rtksort.txt",
                    "offset_file": "2025-10-28_beacon_25Hz_offset.txt",
                    "no_plot": True,
                    "force_recompute": False,
                    "start_datetime": datetime(2026, 3, 5, 14, 0, 0),
                    "end_datetime": datetime(2026, 3, 5, 13, 0, 0),
                },
            )()
            self.assertEqual(sdn.main(), 2)
        finally:
            sdn.parse_args = old_parse_args

    def test_main_with_plot_and_missing_positions(self) -> None:
        """Main should execute plotting branch and warning branch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "Trigger_plot.yaml"
            det_path = root / "det.txt"

            data = make_sample_data()
            data["evtC"] = {
                "event_number": 102,
                "datetime": "2026-03-05T12:30:46",
                "time": {"999": 3000, "101": 3010},
            }
            yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            det_path.write_text("101 0 0 0\n102 3 0 0\n103 0 4 0\n", encoding="utf-8")

            old_parse_args = sdn.parse_args
            try:
                sdn.parse_args = lambda: type(
                    "Args",
                    (),
                    {
                        "yaml_file": str(yaml_path),
                        "det_pos": str(det_path),
                        "offset_file": str(root / "offset.txt"),
                        "no_plot": False,
                        "force_recompute": False,
                    },
                )()
                exit_code = sdn.main()
            finally:
                sdn.parse_args = old_parse_args

            self.assertEqual(exit_code, 0)
            self.assertTrue(
                (root / "Trigger_plot_du_pair_delta_distribution.png").exists()
            )
            self.assertTrue(
                (root / "Trigger_plot_observed_expected_ratio.png").exists()
            )
            self.assertTrue((root / "Trigger_plot_du_pair_delta_vs_time.png").exists())
            self.assertTrue(
                (root / "Trigger_plot_observed_expected_ratio_vs_time.png").exists()
            )
            self.assertTrue((root / "Trigger_plot_event_du_pair_count_vs_time.png").exists())
            self.assertTrue((root / "Trigger_plot_event_du_pair_count_hist.png").exists())
            self.assertTrue((root / "Trigger_plot_event_max_ratio_vs_time.png").exists())
            self.assertTrue((root / "Trigger_plot_event_max_ratio_hist.png").exists())
            self.assertTrue((root / "Trigger_plot_theoretical_delta_vs_time.png").exists())
            self.assertTrue((root / "Trigger_plot_theoretical_delta_hist.png").exists())

    def test_main_no_plot_outputs_csv(self) -> None:
        """Main should emit CSV outputs in --no-plot mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "Trigger_test.yaml"
            det_path = root / "det.txt"

            yaml_path.write_text(
                yaml.safe_dump(make_sample_data(), sort_keys=False),
                encoding="utf-8",
            )
            det_path.write_text(
                "101 0 0 0\n102 3 0 0\n103 0 4 0\n",
                encoding="utf-8",
            )

            # Simulate CLI arguments
            old_parse_args = sdn.parse_args
            try:
                sdn.parse_args = lambda: type(
                    "Args",
                    (),
                    {
                        "yaml_file": str(yaml_path),
                        "det_pos": str(det_path),
                        "offset_file": str(root / "offset.txt"),
                        "no_plot": True,
                        "force_recompute": False,
                    },
                )()
                exit_code = sdn.main()
            finally:
                sdn.parse_args = old_parse_args

            self.assertEqual(exit_code, 0)

            distribution_csv = root / "Trigger_test_du_pair_delta_distribution.csv"
            self.assertTrue(distribution_csv.exists())

            with distribution_csv.open("r", encoding="utf-8") as file_obj:
                rows = list(csv.DictReader(file_obj))
            self.assertEqual(len(rows), 4)
            self.assertNotIn("event_key", rows[0])
            self.assertIn("observed_delta_ns", rows[0])
            self.assertIn("theoretical_delta_ns", rows[0])

            theoretical_cache = root / "det_du_pair_theoretical.csv"
            self.assertTrue(theoretical_cache.exists())

    def test_main_uses_existing_distribution_csv_fast_path(self) -> None:
        """Should use existing distribution CSV directly without reading YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "Trigger_cached.yaml"
            distribution_csv = root / "Trigger_cached_du_pair_delta_distribution.csv"

            with distribution_csv.open("w", newline="", encoding="utf-8") as file_obj:
                writer = csv.writer(file_obj)
                writer.writerow(
                    [
                        "event_number",
                        "event_time",
                        "du_a",
                        "du_b",
                        "observed_delta_ns",
                        "observed_abs_delta_ns",
                        "theoretical_delta_ns",
                        "distance_m",
                        "event_datetime",
                    ]
                )
                writer.writerow(
                    [
                        100,
                        1771027344,
                        "101",
                        "102",
                        10.0,
                        10.0,
                        8.0,
                        3.0,
                        "2026-03-05T12:30:44",
                    ]
                )

            distribution_meta = sdn.distribution_meta_path(distribution_csv)
            meta = sdn.build_distribution_cache_meta(
                yaml_path,
                root / "det.txt",
                root / "offset.txt",
                None,
                None,
            )
            sdn.write_distribution_cache_meta(distribution_meta, meta)

            old_parse_args = sdn.parse_args
            try:
                sdn.parse_args = lambda: type(
                    "Args",
                    (),
                    {
                        "yaml_file": str(yaml_path),
                        "det_pos": str(root / "det.txt"),
                        "offset_file": str(root / "offset.txt"),
                        "no_plot": False,
                        "force_recompute": False,
                    },
                )()
                exit_code = sdn.main()
            finally:
                sdn.parse_args = old_parse_args

            self.assertEqual(exit_code, 0)
            self.assertTrue((root / "Trigger_cached_du_pair_delta_distribution.png").exists())
            self.assertTrue((root / "Trigger_cached_observed_expected_ratio.png").exists())
            self.assertTrue((root / "Trigger_cached_du_pair_delta_vs_time.png").exists())
            self.assertTrue(
                (root / "Trigger_cached_observed_expected_ratio_vs_time.png").exists()
            )
            self.assertTrue((root / "Trigger_cached_event_du_pair_count_vs_time.png").exists())
            self.assertTrue((root / "Trigger_cached_event_du_pair_count_hist.png").exists())
            self.assertTrue((root / "Trigger_cached_event_max_ratio_vs_time.png").exists())
            self.assertTrue((root / "Trigger_cached_event_max_ratio_hist.png").exists())
            self.assertTrue((root / "Trigger_cached_theoretical_delta_vs_time.png").exists())
            self.assertTrue((root / "Trigger_cached_theoretical_delta_hist.png").exists())

    def test_main_cache_path_no_plot_branch(self) -> None:
        """Should hit cache fast-path with no-plot branch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "Trigger_cached_noplot.yaml"
            distribution_csv = (
                root / "Trigger_cached_noplot_du_pair_delta_distribution.csv"
            )

            with distribution_csv.open("w", newline="", encoding="utf-8") as file_obj:
                writer = csv.writer(file_obj)
                writer.writerow(
                    [
                        "event_number",
                        "event_time",
                        "du_a",
                        "du_b",
                        "observed_delta_ns",
                        "observed_abs_delta_ns",
                        "theoretical_delta_ns",
                        "distance_m",
                        "event_datetime",
                    ]
                )
                writer.writerow(
                    [100, 1771027344, "101", "102", 10.0, 10.0, 8.0, 3.0, ""]
                )

            distribution_meta = sdn.distribution_meta_path(distribution_csv)
            meta = sdn.build_distribution_cache_meta(
                yaml_path,
                root / "det.txt",
                root / "offset.txt",
                None,
                None,
            )
            sdn.write_distribution_cache_meta(distribution_meta, meta)

            old_parse_args = sdn.parse_args
            try:
                sdn.parse_args = lambda: type(
                    "Args",
                    (),
                    {
                        "yaml_file": str(yaml_path),
                        "det_pos": str(root / "det.txt"),
                        "offset_file": str(root / "offset.txt"),
                        "no_plot": True,
                        "force_recompute": False,
                    },
                )()
                exit_code = sdn.main()
            finally:
                sdn.parse_args = old_parse_args

            self.assertEqual(exit_code, 0)

    def test_distribution_cache_meta_drift_fallback_recompute(self) -> None:
        """Should recompute when YAML/det-pos/offset/datetime signatures drift."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "Trigger_meta_drift.yaml"
            det_path = root / "det.txt"
            offset_path = root / "offset.txt"
            distribution_csv = root / "Trigger_meta_drift_du_pair_delta_distribution.csv"
            distribution_meta = sdn.distribution_meta_path(distribution_csv)

            yaml_path.write_text(
                yaml.safe_dump(make_sample_data(), sort_keys=False),
                encoding="utf-8",
            )
            det_path.write_text(
                "101 0 0 0\n102 3 0 0\n103 0 4 0\n",
                encoding="utf-8",
            )
            offset_path.write_text("101, 0, 1\n", encoding="utf-8")

            def write_stale_distribution_cache() -> None:
                with distribution_csv.open("w", newline="", encoding="utf-8") as file_obj:
                    writer = csv.writer(file_obj)
                    writer.writerow(
                        [
                            "event_number",
                            "event_time",
                            "du_a",
                            "du_b",
                            "observed_delta_ns",
                            "observed_abs_delta_ns",
                            "theoretical_delta_ns",
                            "distance_m",
                            "event_datetime",
                        ]
                    )
                    writer.writerow(
                        [999, 1771027344, "101", "102", 1.0, 1.0, 1.0, 1.0, "2026-03-05T12:30:44"]
                    )

            def run_main(start_dt=None, end_dt=None) -> int:
                old_parse_args = sdn.parse_args
                try:
                    sdn.parse_args = lambda: type(
                        "Args",
                        (),
                        {
                            "yaml_file": str(yaml_path),
                            "det_pos": str(det_path),
                            "offset_file": str(offset_path),
                            "no_plot": True,
                            "force_recompute": False,
                            "start_datetime": start_dt,
                            "end_datetime": end_dt,
                        },
                    )()
                    return sdn.main()
                finally:
                    sdn.parse_args = old_parse_args

            # Case 1: YAML changed after meta written -> fallback recompute
            write_stale_distribution_cache()
            sdn.write_distribution_cache_meta(
                distribution_meta,
                sdn.build_distribution_cache_meta(
                    yaml_path,
                    det_path,
                    offset_path,
                    None,
                    None,
                ),
            )
            yaml_path.write_text(
                yaml.safe_dump(make_sample_data(), sort_keys=False) + "\n# changed\n",
                encoding="utf-8",
            )
            self.assertEqual(run_main(), 0)
            with distribution_csv.open("r", encoding="utf-8") as file_obj:
                rows = list(csv.DictReader(file_obj))
            self.assertEqual(len(rows), 4)
            self.assertNotEqual(int(rows[0]["event_number"]), 999)

            # Case 2: det-pos or offset changed after meta -> fallback recompute
            write_stale_distribution_cache()
            sdn.write_distribution_cache_meta(
                distribution_meta,
                sdn.build_distribution_cache_meta(
                    yaml_path,
                    det_path,
                    offset_path,
                    None,
                    None,
                ),
            )
            det_path.write_text(
                "101 0 0 0\n102 30 0 0\n103 0 40 0\n",
                encoding="utf-8",
            )
            offset_path.write_text("101, 1, 1\n", encoding="utf-8")
            self.assertEqual(run_main(), 0)
            with distribution_csv.open("r", encoding="utf-8") as file_obj:
                rows = list(csv.DictReader(file_obj))
            self.assertEqual(len(rows), 4)
            self.assertNotEqual(int(rows[0]["event_number"]), 999)

            # Case 3: datetime filter changed after meta -> fallback recompute
            write_stale_distribution_cache()
            sdn.write_distribution_cache_meta(
                distribution_meta,
                sdn.build_distribution_cache_meta(
                    yaml_path,
                    det_path,
                    offset_path,
                    None,
                    None,
                ),
            )
            start_dt = datetime(2026, 3, 5, 12, 30, 45)
            end_dt = datetime(2026, 3, 5, 12, 30, 45)
            self.assertEqual(run_main(start_dt, end_dt), 0)
            with distribution_csv.open("r", encoding="utf-8") as file_obj:
                rows = list(csv.DictReader(file_obj))
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]["event_number"]), 101)

    def test_main_force_recompute_branch(self) -> None:
        """Should ignore cached distribution when force recompute is enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "Trigger_force.yaml"
            det_path = root / "det.txt"
            distribution_csv = root / "Trigger_force_du_pair_delta_distribution.csv"

            yaml_path.write_text(
                yaml.safe_dump(make_sample_data(), sort_keys=False),
                encoding="utf-8",
            )
            det_path.write_text(
                "101 0 0 0\n102 3 0 0\n103 0 4 0\n",
                encoding="utf-8",
            )
            distribution_csv.write_text("stale", encoding="utf-8")

            old_parse_args = sdn.parse_args
            try:
                sdn.parse_args = lambda: type(
                    "Args",
                    (),
                    {
                        "yaml_file": str(yaml_path),
                        "det_pos": str(det_path),
                        "offset_file": str(root / "offset.txt"),
                        "no_plot": True,
                        "force_recompute": True,
                    },
                )()
                exit_code = sdn.main()
            finally:
                sdn.parse_args = old_parse_args

            self.assertEqual(exit_code, 0)
            with distribution_csv.open("r", encoding="utf-8") as file_obj:
                first_line = file_obj.readline().strip()
            self.assertIn("event_number", first_line)

    def test_load_or_build_theoretical_rows_loads_existing_cache(self) -> None:
        """Should load cache directly when shared theoretical file already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            det_pos_path = root / "det.txt"
            det_pos_path.write_text("101 0 0 0\n102 3 0 0\n", encoding="utf-8")

            expected_rows = [make_expected_row(("101", "102", 3.0, 10.0))]
            cache_path = sdn.theoretical_cache_path(det_pos_path)
            sdn.write_theoretical_cache(cache_path, expected_rows)

            detector_positions = {
                "101": np.array([0.0, 0.0, 0.0]),
                "102": np.array([3.0, 0.0, 0.0]),
            }
            loaded = sdn.load_or_build_theoretical_rows(det_pos_path, detector_positions)
            self.assertEqual(loaded, expected_rows)

    def test_main_module_entrypoint(self) -> None:
        """Should execute __main__ entrypoint and raise SystemExit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "Trigger_main.yaml"
            det_path = root / "det.txt"

            yaml_path.write_text(yaml.safe_dump(make_sample_data(), sort_keys=False), encoding="utf-8")
            det_path.write_text("101 0 0 0\n102 3 0 0\n103 0 4 0\n", encoding="utf-8")

            old_argv = sys.argv
            try:
                sys.argv = [
                    "stats_du_pairs.py",
                    str(yaml_path),
                    "--det-pos",
                    str(det_path),
                    "--no-plot",
                ]
                with self.assertRaises(SystemExit):
                    runpy.run_path(str(Path(sdn.__file__)), run_name="__main__")
            finally:
                sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
