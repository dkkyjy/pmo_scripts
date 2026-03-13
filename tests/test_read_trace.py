"""Unit tests for readroot/read_trace.py."""

from __future__ import annotations

import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

import loop
import readroot.read_trace as rt


class DummyTree:
    """Dummy uproot tree object for arrays access."""

    def __init__(self, payload: dict) -> None:
        """Store payload returned by arrays."""
        self._payload = payload

    def arrays(self, _fields: list[str], library: str = "np") -> dict:
        """Return predefined arrays payload."""
        if library != "np":
            raise ValueError("library should be np in this test")
        return self._payload


class DummyRootFile:
    """Dummy uproot file object exposing keys and item access."""

    def __init__(self, keys: list[str], trees: dict[str, DummyTree]) -> None:
        """Initialize with key list and mapping to trees."""
        self._keys = keys
        self._trees = trees

    def keys(self) -> list[str]:
        """Return ROOT keys."""
        return self._keys

    def __getitem__(self, key: str) -> DummyTree:
        """Return tree by key."""
        return self._trees[key]


class FakeArray:
    """Simple object exposing .tolist() like numpy arrays."""

    def __init__(self, value) -> None:
        """Keep raw test value."""
        self._value = value

    def tolist(self):
        """Return raw value for tests."""
        return self._value


class TestReadTrace(unittest.TestCase):
    """Tests for read_trace helpers and workflow."""

    def test_parse_args_defaults_and_alias(self) -> None:
        """Parse positional/optional CLI args correctly."""
        args = rt.parse_args(["prog", "input.root"])
        self.assertEqual(args.file_path, "input.root")
        self.assertEqual(args.left, 0)
        self.assertIsNone(args.right)
        self.assertEqual(args.date, "")
        self.assertEqual(args.out_dir_base, "../Reco_Dir")

        args_alias = rt.parse_args(
            [
                "prog",
                "input.root",
                "--left",
                "10",
                "--right",
                "20",
                "--date",
                "2026/03/04",
                "--out-dir-base",
                "/tmp/out",
            ]
        )
        self.assertEqual(args_alias.left, 10)
        self.assertEqual(args_alias.right, 20)
        self.assertEqual(args_alias.date, "2026/03/04")
        self.assertEqual(args_alias.out_dir_base, "/tmp/out")

    def test_calculate_trace_max_values_and_xy_combined(self) -> None:
        """Compute per-channel max values and XY combined max values."""
        trace_list = [[[1, -2, 3], [4, -5]], [[]]]
        result = rt.calculate_trace_max_values(trace_list, 0, None)
        self.assertEqual(result[0], [3, 5])
        self.assertEqual(result[1], [10])

        trace_x = [[[3, 0], [0, 4]], [[]]]
        trace_y = [[[4, 0], [0, 3]], [[]]]
        xy_result = rt.calculate_xy_combined_max_values(trace_x, trace_y, 0, None)
        self.assertEqual(xy_result[0], [5, 5])
        self.assertEqual(xy_result[1], [10])

    def test_read_file_du_time_ns_success_and_missing_tree(self) -> None:
        """Read expected fields and raise KeyError when teventadc is missing."""
        payload = {
            "run_number": FakeArray([101]),
            "event_number": FakeArray([1]),
            "du_id": FakeArray([[11, 12]]),
            "gps_time": FakeArray([[20260304, 12, 100]]),
            "du_nanoseconds": FakeArray([[1000, 1001]]),
            "trace_0": FakeArray([[[1, -2], [3, -4]]]),
            "trace_1": FakeArray([[[3, 0], [0, 4]]]),
            "trace_2": FakeArray([[[4, 0], [0, 3]]]),
            "trace_3": FakeArray([[[1, 1], [2, 2]]]),
        }
        root_file = DummyRootFile(["teventadc;1"], {"teventadc;1": DummyTree(payload)})

        with patch.object(rt.uproot, "open", return_value=root_file):
            result = rt.read_file_du_time_ns("sample.root", 0, None)

        self.assertEqual(result[0], [101])
        self.assertEqual(result[1], [1])
        self.assertEqual(result[2], [[11, 12]])
        self.assertEqual(result[5], [[2, 4]])
        self.assertEqual(result[9], [[5, 5]])

        no_tree_file = DummyRootFile(["other;1"], {"other;1": DummyTree({})})
        with patch.object(rt.uproot, "open", return_value=no_tree_file):
            with self.assertRaises(KeyError):
                rt.read_file_du_time_ns("sample.root", 0, None)

    def test_cal_dict_du_ns_and_skip_invalid_first_ns(self) -> None:
        """Build event payload and skip rows whose first nanosecond is invalid."""
        data_dict: dict = {}
        rt.cal_dict_du_ns(
            run_number_list=[101, 101],
            event_number_list=[5001, 5002],
            du_id_list=[[11, 12], [13]],
            gps_time_list=[[20260304, 12, 1772000000], [20260304, 13, 1772000001]],
            du_nanosecond_list=[[100, 120], [0]],
            data_dict=data_dict,
            trace_adc_maxvalue_list=[[7, 8], [9]],
            file_path="Trigger_x.root",
        )

        self.assertIn("5001", data_dict)
        self.assertNotIn("5002", data_dict)
        payload = data_dict["5001"]
        self.assertEqual(payload["time"], {"11": [100], "12": [120]})
        self.assertEqual(payload["signal"], {"11": [7], "12": [8]})
        self.assertEqual(payload["datetime"], "2026-03-04T00:00:12")

    def test_cal_dict_du_ns_preserves_multiple_triggers_per_du(self) -> None:
        """Same DU should keep all nanosecond/amplitude samples as lists."""
        data_dict: dict = {}
        rt.cal_dict_du_ns(
            run_number_list=[101],
            event_number_list=[6001],
            du_id_list=[[11, 11, 12]],
            gps_time_list=[[20260304, 12, 1772000000]],
            du_nanosecond_list=[[100, 130, 140]],
            data_dict=data_dict,
            trace_adc_maxvalue_list=[[7, 9, 8]],
            file_path="Trigger_dup.root",
        )

        payload = data_dict["6001"]
        self.assertEqual(payload["time"], {"11": [100, 130], "12": [140]})
        self.assertEqual(payload["signal"], {"11": [7, 9], "12": [8]})
        self.assertEqual(payload["du_id"], ["11", "11", "12"])

    def test_process_single_file_and_write_outputs(self) -> None:
        """Populate all channel dicts and write channel output YAML files."""
        read_result = (
            [101],
            [5001],
            [[11]],
            [[20260304, 120000, 1772000000]],
            [[100]],
            [[10]],
            [[11]],
            [[12]],
            [[13]],
            [[14]],
        )

        dict_f: dict = {}
        dict_x: dict = {}
        dict_y: dict = {}
        dict_z: dict = {}
        dict_xy: dict = {}

        with patch.object(rt, "read_file_du_time_ns", return_value=read_result):
            rt.process_single_file(
                "/tmp/abc.root",
                0,
                10,
                dict_f,
                dict_x,
                dict_y,
                dict_z,
                dict_xy,
            )

        self.assertIn("5001", dict_f)
        self.assertEqual(dict_xy["5001"]["signal"]["11"], [14])

        with tempfile.TemporaryDirectory() as tmpdir:
            rt.write_outputs(
                dict_f,
                dict_x,
                dict_y,
                dict_z,
                dict_xy,
                tmpdir,
                "Trigger_test",
            )
            expected_files = [
                "Trigger_test_F.yaml",
                "Trigger_test_X.yaml",
                "Trigger_test_Y.yaml",
                "Trigger_test_Z.yaml",
                "Trigger_test_XY.yaml",
            ]
            for name in expected_files:
                self.assertTrue((Path(tmpdir) / name).exists())

            loaded = yaml.safe_load((Path(tmpdir) / "Trigger_test_F.yaml").read_text())
            self.assertIn("5001", loaded)

    def test_process_root_file_and_main_status_code(self) -> None:
        """Return True/False from process_root_file and map to shell status in main."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_base = Path(tmpdir) / "Reco_Dir"
            ok = rt.process_root_file(
                "Trigger_test.root",
                0,
                512,
                "2026/03/04",
                str(out_base),
            )
            self.assertTrue(ok)
            self.assertTrue((out_base / "2026/03/04" / "Trigger_test_F.yaml").exists())

        with patch.object(rt, "mkdir", side_effect=RuntimeError("mkdir failed")):
            self.assertFalse(rt.process_root_file("a.root", 0, 1, "d", "/tmp/out"))

        args = rt.argparse.Namespace(
            file_path="a.root",
            left=0,
            right=512,
            date="2026/03/04",
            out_dir_base="/tmp/out",
        )
        with patch.object(rt, "parse_args", return_value=args):
            with patch.object(rt, "process_root_file", return_value=True):
                self.assertEqual(rt.main(["prog"]), 0)
            with patch.object(rt, "process_root_file", return_value=False):
                self.assertEqual(rt.main(["prog"]), 2)

    def test_loop_make_readtrace_command_uses_new_script(self) -> None:
        """Build read-trace command with renamed script path."""
        cmd = loop.make_readtrace_command("a.root", "2026/03/04", "../Reco_Dir")
        self.assertEqual(cmd[1], "readroot/read_trace.py")

    def test_mkdir_existing_directory_branch(self) -> None:
        """Calling mkdir twice should hit existing-directory branch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "already_exists"
            rt.mkdir(str(target))
            rt.mkdir(str(target))
            self.assertTrue(target.exists())

    def test_module_entrypoint_executes_main(self) -> None:
        """Executing module as __main__ should run entrypoint line."""
        old_argv = sys.argv
        try:
            # Missing required file_path argument makes parse_args exit early.
            sys.argv = ["read_trace.py"]
            with self.assertRaises(SystemExit):
                runpy.run_path(str(Path(rt.__file__)), run_name="__main__")
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
