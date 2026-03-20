"""Unit tests for stats_trigger_yaml.py.

This suite validates:
1) New-format parsing using gps_time and time.
2) Core aggregation helpers and CSV writers.
3) Main workflow behavior for success/error paths.
"""

from __future__ import annotations

import csv
import argparse
import runpy
import tempfile
import unittest
from unittest import mock
from datetime import datetime
from pathlib import Path

from stats.stats_trigger import (
    EVENT_RECORD_FIELDS,
    parse_event_records,
    read_event_csv,
)

import yaml

import stats.stats_trigger as st


def make_new_format_data() -> dict:
    """Create sample events using the new payload format."""
    return {
        "28862": {
            "datetime": "2026-02-14T00:02:24",
            "gps_time": 1771027344,
            "du_id": ["1014", "1088"],
            "time": {"1014": 29681922, "1088": 29681892},
            "event_number": 28862,
            "index": 0,
        },
        "28863": {
            "datetime": "2026-02-14T00:02:25",
            "gps_time": 1771027345,
            "du_id": None,
            "time": {"1071": 46689808, "1081": 46680812},
            "event_number": 28863,
            "index": 1,
        },
        "28864": {
            "datetime": "2026-02-14T00:02:25",
            "gps_time": 1771027345,
            "du_id": ["1081"],
            "time": {"1081": 46690000},
            "event_number": 28864,
            "index": 2,
        },
    }


class TestStatsTriggerYamlUnit(unittest.TestCase):
    """Unit tests for stats_trigger helpers and CLI workflow."""

    def test_parse_event_second_helpers(self) -> None:
        """Parse event second strictly from payload gps_time."""
        payload = {"gps_time": 1771027344}
        self.assertEqual(
            st.parse_event_second_from_payload("28862", payload),
            1771027344,
        )

        with self.assertRaises(ValueError):
            st.parse_event_second_from_payload("28863", {"gps_time": "invalid"})

        with self.assertRaises(ValueError):
            st.parse_event_second_from_payload("28864", {})

    def test_get_du_ns_map(self) -> None:
        """Read DU map strictly from time."""
        payload = {"time": {"101": 1}}
        self.assertEqual(st.get_du_ns_map(payload), {"101": 1})
        self.assertEqual(st.get_du_ns_map({"time": {"2": 5}}), {"2": 5})
        self.assertEqual(st.get_du_ns_map({"time": []}), {})

    def test_named_default_row_setitem_branches(self) -> None:
        """Cover NamedDefaultRow __setitem__ for named keys."""
        row = st.make_named_row(("a", "b"), a=1, b=2)
        row["a"] = 3
        row["b"] = 4
        self.assertEqual(row["a"], 3)
        self.assertEqual(row["b"], 4)

    def test_named_default_row_disallow_positional_access(self) -> None:
        """Disallow positional get/set and tuple/list compatibility."""
        row = st.make_named_row(("a", "b"), a=1, b=2)

        with self.assertRaises(TypeError):
            _ = row[0]

        with self.assertRaises(TypeError):
            row[0] = 5

        self.assertNotEqual(row, (1, 2))
        self.assertNotEqual(row, [1, 2])

    def test_parse_event_records_new_format(self) -> None:
        """Build records from new format and keep expected ordering/fields."""
        data = make_new_format_data()
        records = st.parse_event_records(data)

        self.assertEqual(len(records), 3)
        first = records[0]
        self.assertEqual(first["event_second"], 1771027344)
        self.assertEqual(first["du_count"], 2)
        self.assertEqual(first["event_datetime"], "2026-02-14T00:02:24")

        second = records[1]
        self.assertEqual(second["event_second"], 1771027345)
        self.assertEqual(second["du_count"], 2)

    def test_parse_event_records_requires_gps_time(self) -> None:
        """Fail when gps_time is missing in payload."""
        data = {
            "28862": {
                "du_id": ["1"],
                "time": {"1": 10},
                "event_number": 1,
                "index": 0,
            }
        }
        with self.assertRaises(ValueError):
            st.parse_event_records(data)

    def test_empty_record_parsing_branch(self) -> None:
        """Cover empty record parsing branch."""
        self.assertEqual(st.parse_event_records({}), [])

    def test_build_rate_and_du_aggregates(self) -> None:
        """Compute per-second and per-DU aggregates correctly."""
        data = make_new_format_data()
        records = st.parse_event_records(data)

        rates = st.build_rate_per_second(records)
        rate_rows = [
            (row["event_second"], row["event_count"], row["event_rate_hz"])
            for row in rates
        ]
        self.assertEqual(rate_rows, [(1771027344, 1, 1.0), (1771027345, 2, 2.0)])

        du_rates = st.build_du_trigger_rate(data, records)
        du_rate_dict = {
            row["du_id"]: (row["trigger_count"], row["trigger_rate_hz"])
            for row in du_rates
        }
        self.assertEqual(du_rate_dict["1081"][0], 2)
        self.assertEqual(du_rate_dict["1014"][0], 1)

    def test_adjacent_time_delta_distribution(self) -> None:
        """Build adjacent event-time deltas and grouped distribution rows."""
        data = make_new_format_data()
        records = st.parse_event_records(data)
        deltas = st.build_adjacent_time_deltas(records)
        self.assertEqual(deltas, [1, 0])

        distribution = st.build_adjacent_time_delta_distribution(deltas)
        distribution_rows = [
            (row["delta_second"], row["event_pair_count"]) for row in distribution
        ]
        self.assertEqual(distribution_rows, [(0, 1), (1, 1)])

    def test_per_second_du_helpers(self) -> None:
        """Create per-second DU counters and flattened rows."""
        data = make_new_format_data()
        per_second = st.build_du_counts_per_second(data)
        self.assertEqual(per_second[1771027345]["1081"], 2)

        total_rows = st.build_total_du_trigger_per_second_rows(per_second)
        total_row_pairs = [
            (row["event_second"], row["total_du_trigger_count"])
            for row in total_rows
        ]
        self.assertEqual(total_row_pairs, [(1771027344, 2), (1771027345, 3)])

        rows = st.build_du_rate_per_second_rows(per_second)
        self.assertTrue(
            any(
                row["event_second"] == 1771027344 and row["du_id"] == "1014"
                for row in rows
            )
        )
        self.assertTrue(
            any(
                row["event_second"] == 1771027345 and row["du_id"] == "1081"
                for row in rows
            )
        )

    def test_window_averages(self) -> None:
        """Compute window averages with expected counts and rates."""
        data = make_new_format_data()
        records = st.parse_event_records(data)
        per_second = st.build_du_counts_per_second(data)

        event_rows, du_rows = st.build_window_averages(records, per_second, 2)
        self.assertEqual(len(event_rows), 1)
        self.assertEqual(event_rows[0]["event_count"], 3)
        self.assertAlmostEqual(event_rows[0]["avg_event_rate_hz"], 1.5)
        self.assertTrue(
            any(
                row["du_id"] == "1081" and row["trigger_count"] == 2
                for row in du_rows
            )
        )

        total_rows = st.build_total_du_trigger_per_second_rows(per_second)
        avg_total_rows = st.build_avg_total_du_trigger_rows(total_rows, 2)
        self.assertEqual(len(avg_total_rows), 1)
        self.assertEqual(avg_total_rows[0]["total_du_trigger_count"], 5)
        self.assertAlmostEqual(
            avg_total_rows[0]["avg_total_du_trigger_rate_hz"],
            2.5,
        )

    def test_write_event_csv_columns(self) -> None:
        """Write event CSV with event_number first and no event_key column."""
        records = st.parse_event_records(make_new_format_data())
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "events.csv"
            st.write_event_csv(csv_path, records)

            with csv_path.open("r", encoding="utf-8") as file_obj:
                reader = csv.DictReader(file_obj)
                fieldnames = reader.fieldnames or []
                rows = list(reader)

        self.assertEqual(len(rows), 3)
        self.assertEqual(fieldnames[0], "event_number")
        self.assertNotIn("event_key", fieldnames)
        self.assertNotIn("index", fieldnames)
        self.assertIn("event_datetime", rows[0])
        self.assertNotIn("event_date", rows[0])
        self.assertNotIn("event_time", rows[0])
        self.assertEqual(rows[0]["event_datetime"], "2026-02-14T00:02:24")

    def test_write_avg_csv_and_empty_helpers(self) -> None:
        """Cover avg helper branches."""
        self.assertEqual(st.build_du_trigger_rate({}, []), [])
        self.assertEqual(st.build_window_averages([], {}, 60), ([], []))

    def test_plot_functions_non_empty_and_empty(self) -> None:
        """Cover all plotting functions for both normal and empty branches."""
        data = make_new_format_data()
        records = st.parse_event_records(data)
        rates = st.build_rate_per_second(records)
        du_rates = st.build_du_trigger_rate(data, records)
        per_second = st.build_du_counts_per_second(data)
        total_du_rows = st.build_total_du_trigger_per_second_rows(per_second)
        avg_event_rows, avg_du_rows = st.build_window_averages(records, per_second, 2)
        avg_total_rows = st.build_avg_total_du_trigger_rows(total_du_rows, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            st.plot_du_count_histogram(root / "hist.png", records)
            st.plot_event_du_count_over_time(root / "event_du_time.png", records)
            st.plot_rate_line(root / "rate.png", rates)
            st.plot_du_trigger_rate(root / "du.png", du_rates)
            st.plot_total_du_trigger_per_second(
                root / "total_du_rate.png",
                st.build_total_du_trigger_per_second_rows(per_second),
            )
            st.plot_du_rate_per_second_heatmap(root / "heat.png", per_second)
            st.plot_avg_event_rate(root / "avg_event.png", avg_event_rows, 2)
            st.plot_avg_total_du_trigger_rate(
                root / "avg_total_du.png", avg_total_rows, 2
            )
            st.plot_avg_du_rate_heatmap(root / "avg_du.png", avg_du_rows, 2)
            st.plot_avg_du_rate_topn_lines(root / "topn.png", avg_du_rows, 2, 2)

            self.assertTrue((root / "hist.png").exists())
            self.assertTrue((root / "event_du_time.png").exists())
            self.assertTrue((root / "rate.png").exists())
            self.assertTrue((root / "du.png").exists())
            self.assertTrue((root / "total_du_rate.png").exists())
            self.assertTrue((root / "heat.png").exists())
            self.assertTrue((root / "avg_event.png").exists())
            self.assertTrue((root / "avg_total_du.png").exists())
            self.assertTrue((root / "avg_du.png").exists())
            self.assertTrue((root / "topn.png").exists())

            st.plot_du_count_histogram(root / "hist_empty.png", [])
            st.plot_event_du_count_over_time(root / "event_du_time_empty.png", [])
            st.plot_rate_line(root / "rate_empty.png", [])
            st.plot_du_trigger_rate(root / "du_empty.png", [])
            st.plot_total_du_trigger_per_second(root / "total_du_rate_empty.png", [])
            st.plot_du_rate_per_second_heatmap(root / "heat_empty.png", {})
            st.plot_avg_event_rate(root / "avg_event_empty.png", [], 2)
            st.plot_avg_total_du_trigger_rate(
                root / "avg_total_du_empty.png", [], 2
            )
            st.plot_avg_du_rate_heatmap(root / "avg_du_empty.png", [], 2)
            st.plot_avg_du_rate_topn_lines(root / "topn_empty.png", [], 2, 2)
            st.plot_avg_du_rate_topn_lines(root / "topn_zero.png", avg_du_rows, 2, 0)

    def test_plot_datetime_tick_branches(self) -> None:
        """Cover datetime tick label branches in helper and heatmap plots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            fig1, ax1 = st.plt.subplots()
            st.apply_gps_datetime_ticks(ax1, [1], {1: ""}, max_ticks=0)
            st.apply_gps_datetime_ticks(ax1, [1], {1: ""}, max_ticks=1)
            st.plt.close(fig1)

            per_second_single = {100: st.Counter({"101": 1})}
            st.plot_du_rate_per_second_heatmap(
                root / "heat_one_second.png",
                per_second_single,
                {},
            )
            self.assertTrue((root / "heat_one_second.png").exists())

            avg_rows = [
                st.make_named_row(
                    st.AVG_DU_FIELDS,
                    window_start_second=100,
                    window_end_second_exclusive=101,
                    du_id="101",
                    trigger_count=1,
                    avg_trigger_rate_hz=1.0,
                ),
                st.make_named_row(
                    st.AVG_DU_FIELDS,
                    window_start_second=101,
                    window_end_second_exclusive=102,
                    du_id="101",
                    trigger_count=2,
                    avg_trigger_rate_hz=2.0,
                ),
            ]
            st.plot_avg_du_rate_heatmap(root / "avg_du_ticks.png", avg_rows, 1, {})
            self.assertTrue((root / "avg_du_ticks.png").exists())

    def test_topn_no_selected_du_branch(self) -> None:
        """Cover defensive branch when Top-N selection returns no DU IDs."""
        rows = [
            st.make_named_row(
                st.AVG_DU_FIELDS,
                window_start_second=100,
                window_end_second_exclusive=101,
                du_id="101",
                trigger_count=1,
                avg_trigger_rate_hz=1.0,
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "topn_none.png"
            with mock.patch.object(st.Counter, "most_common", return_value=[]):
                st.plot_avg_du_rate_topn_lines(out_path, rows, 1, 1)
            self.assertFalse(out_path.exists())

    def test_parse_args_custom_values(self) -> None:
        """Cover parse_args with functional options only."""
        with mock.patch(
            "sys.argv",
            [
                "stats_trigger.py",
                "input.yaml",
                "--avg-window",
                "2",
                "--topn",
                "4",
                "--no-plot",
            ],
        ):
            args = st.parse_args()
            self.assertEqual(args.yaml_file, "input.yaml")
            self.assertEqual(args.avg_window, 2)
            self.assertEqual(args.topn, 4)
            self.assertTrue(args.no_plot)
            self.assertFalse(args.force_recompute)

    def test_parse_args_force_recompute(self) -> None:
        """Cover force-recompute CLI option parsing."""
        with mock.patch(
            "sys.argv",
            [
                "stats_trigger.py",
                "input.yaml",
                "--force-recompute",
            ],
        ):
            args = st.parse_args()
            self.assertTrue(args.force_recompute)

    def test_parse_args_datetime_range(self) -> None:
        """Cover datetime-range CLI option parsing."""
        with mock.patch(
            "sys.argv",
            [
                "stats_trigger.py",
                "input.yaml",
                "--start-datetime",
                "2026-02-14T00:02:25",
                "--end-datetime",
                "2026-02-14T00:02:26",
            ],
        ):
            args = st.parse_args()
            self.assertEqual(args.start_datetime, datetime(2026, 2, 14, 0, 2, 25))
            self.assertEqual(args.end_datetime, datetime(2026, 2, 14, 0, 2, 26))

    def test_datetime_helper_edge_branches(self) -> None:
        """Cover CLI/payload datetime parsing and filter invalid branches."""
        with self.assertRaises(argparse.ArgumentTypeError):
            st.parse_cli_datetime("2026/02/14 00:00:00")

        self.assertEqual(st.parse_event_datetime({"datetime": ""}), "")
        self.assertEqual(
            st.parse_event_datetime({"datetime": "2026-02-14Tbad"}),
            "",
        )

        event_dt = datetime(2026, 2, 14, 0, 2, 25)
        self.assertFalse(
            st.in_datetime_range(
                event_dt,
                datetime(2026, 2, 14, 0, 2, 0),
                datetime(2026, 2, 14, 0, 2, 24),
            )
        )

        input_data = {
            "1": {"datetime": "2026-02-14T00:02:25", "gps_time": 1771027345},
            "2": {"datetime": "2026-02-14Tbad", "gps_time": 1771027346},
        }
        filtered = st.filter_data_by_datetime_range(
            input_data,
            datetime(2026, 2, 14, 0, 2, 25),
            datetime(2026, 2, 14, 0, 2, 25),
        )
        self.assertEqual(list(filtered.keys()), ["1"])

    def test_event_csv_du_ids_roundtrip_helper(self) -> None:
        """Cover event CSV DU-id mapping helper."""
        data = make_new_format_data()
        records = st.parse_event_records(data)
        event_du_ids_map = st.build_event_du_ids_map(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "events.csv"
            st.write_event_csv(csv_path, records, event_du_ids_map)
            loaded_du_ids_map = st.read_event_csv_du_ids(csv_path)

        self.assertEqual(
            loaded_du_ids_map[str(records[0]["event_number"])],
            event_du_ids_map[str(records[0]["event_number"])],
        )

    def test_adjacent_delta_plot_edge_branches(self) -> None:
        """Cover adjacent-delta plotting branches for empty and fixed-bin input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            st.plot_adjacent_time_delta_histogram(root / "delta_empty.png", [])
            self.assertFalse((root / "delta_empty.png").exists())

            st.plot_adjacent_time_delta_histogram(root / "delta_single_bin.png", [2, 2])
            self.assertTrue((root / "delta_single_bin.png").exists())

    def test_remaining_empty_and_defensive_branches(self) -> None:
        """Cover final defensive/empty helper branches."""
        self.assertEqual(st.deserialize_du_ids(""), [])

        records = st.parse_event_records(make_new_format_data())
        self.assertEqual(st.build_adjacent_time_deltas(records[:1]), [])

        fig, ax = st.plt.subplots()
        with mock.patch("stats.stats_trigger.set", return_value=set()):
            st.apply_gps_datetime_ticks(ax, [1], {1: ""})
        st.plt.close(fig)

    def test_main_invalid_datetime_range(self) -> None:
        """Main should fail when start datetime is later than end datetime."""
        with mock.patch(
            "sys.argv",
            [
                "stats_trigger.py",
                "input.yaml",
                "--start-datetime",
                "2026-02-14T00:03:00",
                "--end-datetime",
                "2026-02-14T00:02:00",
            ],
        ):
            self.assertEqual(st.main(), 2)

    def test_defensive_observation_seconds_branch(self) -> None:
        """Force defensive observation_seconds <= 0 branch via min/max mocking."""
        data = make_new_format_data()
        records = st.parse_event_records(data)
        with mock.patch("builtins.min", return_value=10), mock.patch("builtins.max", return_value=8):
            rows = st.build_du_trigger_rate(data, records)
        self.assertTrue(len(rows) > 0)

    def test_main_error_paths(self) -> None:
        """Return non-zero for missing file and invalid top-level YAML."""
        with mock.patch("sys.argv", ["stats_trigger.py", "/tmp/not_exists.yaml"]):
            self.assertEqual(st.main(), 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            bad_yaml = Path(tmpdir) / "bad.yaml"
            bad_yaml.write_text("- 1\n- 2\n", encoding="utf-8")
            with mock.patch("sys.argv", ["stats_trigger.py", str(bad_yaml)]):
                self.assertEqual(st.main(), 2)

    def test_main_success_no_plot(self) -> None:
        """Run main successfully and keep only one CSV output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "input.yaml"
            yaml_path.write_text(
                yaml.safe_dump(make_new_format_data(), sort_keys=False),
                encoding="utf-8",
            )

            with mock.patch(
                "sys.argv",
                [
                    "stats_trigger.py",
                    str(yaml_path),
                    "--no-plot",
                ],
            ):
                self.assertEqual(st.main(), 0)

            expected = [
                root / "input_event_du_count.csv",
            ]
            for path in expected:
                self.assertTrue(path.exists())

            self.assertFalse((root / "input_rate_per_second.csv").exists())
            self.assertFalse((root / "input_du_trigger_rate.csv").exists())
            self.assertFalse(
                (root / "input_adjacent_time_delta_distribution.csv").exists()
            )
            self.assertFalse((root / "input_du_rate_per_second.csv").exists())

    def test_main_success_no_plot_with_datetime_range(self) -> None:
        """Run main with datetime range and keep only in-range events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "input.yaml"
            yaml_path.write_text(
                yaml.safe_dump(make_new_format_data(), sort_keys=False),
                encoding="utf-8",
            )

            with mock.patch(
                "sys.argv",
                [
                    "stats_trigger.py",
                    str(yaml_path),
                    "--no-plot",
                    "--start-datetime",
                    "2026-02-14T00:02:25",
                    "--end-datetime",
                    "2026-02-14T00:02:25",
                ],
            ):
                self.assertEqual(st.main(), 0)

            event_csv = root / "input_event_du_count.csv"
            self.assertTrue(event_csv.exists())
            with event_csv.open("r", encoding="utf-8") as file_obj:
                rows = list(csv.DictReader(file_obj))

            self.assertEqual(len(rows), 2)
            self.assertTrue(
                all(row["event_datetime"] == "2026-02-14T00:02:25" for row in rows)
            )

    def test_main_success_with_avg_and_plot(self) -> None:
        """Run main with average windows and plotting enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "input.yaml"
            yaml_path.write_text(
                yaml.safe_dump(make_new_format_data(), sort_keys=False),
                encoding="utf-8",
            )

            with mock.patch(
                "sys.argv",
                [
                    "stats_trigger.py",
                    str(yaml_path),
                    "--avg-window",
                    "2",
                    "--topn",
                    "2",
                ],
            ):
                self.assertEqual(st.main(), 0)

            expected = [
                root / "input_event_du_count.csv",
                root / "input_du_count_hist.png",
                root / "input_event_du_count_over_time.png",
                root / "input_rate_per_second.png",
                root / "input_du_trigger_rate.png",
                root / "input_total_du_trigger_per_second.png",
                root / "input_adjacent_time_delta_hist.png",
                root / "input_du_rate_per_second_heatmap.png",
                root / "input_du_trigger_rate_topn.png",
                root / "input_avg_event_rate.png",
                root / "input_avg_total_du_trigger_rate.png",
                root / "input_avg_du_trigger_rate_heatmap.png",
                root / "input_avg_du_trigger_rate_topn.png",
            ]
            for path in expected:
                self.assertTrue(path.exists())

            self.assertFalse((root / "input_avg_event_rate.csv").exists())
            self.assertFalse((root / "input_avg_du_trigger_rate.csv").exists())

    def test_main_success_plot_topn_without_avg_window(self) -> None:
        """Plot Top-N DU curves even when no averaging window is configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "input.yaml"
            yaml_path.write_text(
                yaml.safe_dump(make_new_format_data(), sort_keys=False),
                encoding="utf-8",
            )

            with mock.patch(
                "sys.argv",
                [
                    "stats_trigger.py",
                    str(yaml_path),
                    "--topn",
                    "2",
                ],
            ):
                self.assertEqual(st.main(), 0)

            self.assertTrue((root / "input_du_trigger_rate_topn.png").exists())

    def test_main_cache_without_meta_falls_back_and_requires_yaml(self) -> None:
        """Without cache meta, fallback to YAML path and fail when missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "cache_source.yaml"

            data = make_new_format_data()
            records = st.parse_event_records(data)
            event_du_ids_map = st.build_event_du_ids_map(data)
            st.write_event_csv(
                root / "cache_source_event_du_count.csv",
                records,
                event_du_ids_map,
            )

            with mock.patch(
                "sys.argv",
                [
                    "stats_trigger.py",
                    str(yaml_path),
                    "--no-plot",
                ],
            ):
                self.assertEqual(st.main(), 2)

    def test_main_uses_csv_cache_when_meta_matches(self) -> None:
        """Use existing CSV cache when sidecar meta matches runtime inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "cache_source.yaml"
            yaml_path.write_text(
                yaml.safe_dump(make_new_format_data(), sort_keys=False),
                encoding="utf-8",
            )

            event_csv = root / "cache_source_event_du_count.csv"
            data = make_new_format_data()
            records = st.parse_event_records(data)
            event_du_ids_map = st.build_event_du_ids_map(data)
            st.write_event_csv(event_csv, records, event_du_ids_map)
            st.write_event_cache_meta(
                st.event_cache_meta_path(event_csv),
                st.build_event_cache_meta(yaml_path, None, None),
            )

            with mock.patch(
                "stats.stats_trigger.parse_event_records",
                side_effect=AssertionError("Should use cache and skip YAML parse"),
            ):
                with mock.patch(
                    "sys.argv",
                    [
                        "stats_trigger.py",
                        str(yaml_path),
                        "--no-plot",
                    ],
                ):
                    self.assertEqual(st.main(), 0)

    def test_main_cache_meta_mismatch_recomputes(self) -> None:
        """Cache meta mismatch should fallback to recompute from YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "cache_drift.yaml"
            yaml_path.write_text(
                yaml.safe_dump(make_new_format_data(), sort_keys=False),
                encoding="utf-8",
            )

            event_csv = root / "cache_drift_event_du_count.csv"
            old_data = make_new_format_data()
            old_records = st.parse_event_records(old_data)
            st.write_event_csv(event_csv, old_records, st.build_event_du_ids_map(old_data))
            st.write_event_cache_meta(
                st.event_cache_meta_path(event_csv),
                st.build_event_cache_meta(yaml_path, None, None),
            )

            drift_data = make_new_format_data()
            drift_data["29999"] = {
                "datetime": "2026-02-14T00:02:26",
                "gps_time": 1771027346,
                "du_id": ["1099"],
                "time": {"1099": 12345678},
                "event_number": 29999,
                "index": 3,
            }
            yaml_path.write_text(
                yaml.safe_dump(drift_data, sort_keys=False),
                encoding="utf-8",
            )

            with mock.patch(
                "sys.argv",
                [
                    "stats_trigger.py",
                    str(yaml_path),
                    "--no-plot",
                ],
            ):
                self.assertEqual(st.main(), 0)

            cached_rows = read_event_csv(event_csv)
            self.assertEqual(len(cached_rows), 4)

    def test_main_force_recompute_ignores_csv_cache(self) -> None:
        """Force recompute should not use cache and should require YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "force_source.yaml"

            records = st.parse_event_records(make_new_format_data())
            st.write_event_csv(root / "force_source_event_du_count.csv", records)

            with mock.patch(
                "sys.argv",
                [
                    "stats_trigger.py",
                    str(yaml_path),
                    "--no-plot",
                    "--force-recompute",
                ],
            ):
                self.assertEqual(st.main(), 2)

    def test_main_module_entrypoint(self) -> None:
        """Cover __main__ SystemExit path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_path = root / "input.yaml"
            yaml_path.write_text(
                yaml.safe_dump(make_new_format_data(), sort_keys=False),
                encoding="utf-8",
            )

            with mock.patch(
                "sys.argv",
                [
                    "stats_trigger.py",
                    str(yaml_path),
                    "--no-plot",
                ],
            ):
                with self.assertRaises(SystemExit) as ctx:
                    runpy.run_path(str(Path(st.__file__)), run_name="__main__")
            self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
