"""Unit tests for readroot/read_header.py."""

from __future__ import annotations

import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

import loop
import readroot.read_header as rh


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


class TestReadHeader(unittest.TestCase):
    """Tests for helper functions and CLI workflow."""

    def test_parse_args_defaults_and_alias(self) -> None:
        """Parse positional and optional CLI flags correctly."""
        args = rh.parse_args(["prog", "input.root"])
        self.assertEqual(args.file_path, "input.root")
        self.assertEqual(args.date, "")
        self.assertEqual(args.out_dir_base, "../Reco_Dir")

        args_alias = rh.parse_args(
            [
                "prog",
                "input.root",
                "--date",
                "2026/03/04",
                "--out-dir-base",
                "/tmp/out",
            ]
        )
        self.assertEqual(args_alias.date, "2026/03/04")
        self.assertEqual(args_alias.out_dir_base, "/tmp/out")

    def test_read_file_du_time_ns_success(self) -> None:
        """Read required fields from teventadc tree and return lists."""
        payload = {
            "run_number": FakeArray([101, 101]),
            "event_number": FakeArray([1, 2]),
            "du_id": FakeArray([[11, 12], [13]]),
            "gps_time": FakeArray([[20260304, 12, 100], [20260304, 13, 101]]),
            "du_nanoseconds": FakeArray([[1000, 1001], [1002]]),
        }
        tree = DummyTree(payload)
        root_file = DummyRootFile(["teventadc;1"], {"teventadc;1": tree})

        with patch.object(rh.uproot, "open", return_value=root_file):
            result = rh.read_file_du_time_ns("sample.root")

        self.assertEqual(result[0], [101, 101])
        self.assertEqual(result[1], [1, 2])
        self.assertEqual(result[2], [[11, 12], [13]])
        self.assertEqual(result[3], [[20260304, 12, 100], [20260304, 13, 101]])
        self.assertEqual(result[4], [[1000, 1001], [1002]])

    def test_read_file_du_time_ns_missing_tree(self) -> None:
        """Raise KeyError when teventadc key is missing."""
        root_file = DummyRootFile(["other;1"], {"other;1": DummyTree({})})
        with patch.object(rh.uproot, "open", return_value=root_file):
            with self.assertRaises(KeyError):
                rh.read_file_du_time_ns("sample.root")

    def test_cal_dict_du_ns_populates_event_payload(self) -> None:
        """Build event payload dict using event_number as top-level key."""
        data_dict: dict = {}
        rh.cal_dict_du_ns(
            run_number_list=[101],
            event_number_list=[5001],
            du_id_list=[[11, 12]],
            gps_time_list=[[20260304, 133015, 1772000000]],
            du_nanosecond_list=[[100, 120]],
            data_dict=data_dict,
            file_path="Trigger_x.root",
        )

        self.assertIn("5001", data_dict)
        payload = data_dict["5001"]
        self.assertEqual(payload["run_number"], 101)
        self.assertEqual(payload["event_number"], 5001)
        self.assertEqual(payload["datetime"], "2026-03-04T13:30:15")
        self.assertEqual(payload["gps_time"], 1772000000)
        self.assertEqual(payload["time"], {"11": [100], "12": [120]})
        self.assertEqual(payload["du_id"], ["11", "12"])
        self.assertEqual(payload["file"], "Trigger_x.root")
        self.assertEqual(payload["index"], 0)

    def test_cal_dict_du_ns_keeps_multiple_samples_per_du(self) -> None:
        """Keep repeated DU triggers as list values in time map."""
        data_dict: dict = {}
        rh.cal_dict_du_ns(
            run_number_list=[101],
            event_number_list=[5002],
            du_id_list=[[11, 11, 12]],
            gps_time_list=[[20260304, 133016, 1772000001]],
            du_nanosecond_list=[[100, 130, 120]],
            data_dict=data_dict,
            file_path="Trigger_x.root",
        )

        payload = data_dict["5002"]
        self.assertEqual(payload["time"], {"11": [100, 130], "12": [120]})

    def test_cal_dict_du_ns_skips_empty_du_list(self) -> None:
        """Skip events whose du_id list is empty."""
        data_dict: dict = {}
        rh.cal_dict_du_ns(
            run_number_list=[101],
            event_number_list=[7001],
            du_id_list=[[]],
            gps_time_list=[[20260304, 133017, 1772000002]],
            du_nanosecond_list=[[]],
            data_dict=data_dict,
            file_path="Trigger_x.root",
        )
        self.assertEqual(data_dict, {})

    def test_process_single_file_success_and_error(self) -> None:
        """Populate result dict on success and skip errors gracefully."""
        result: dict = {}

        with patch.object(
            rh,
            "read_file_du_time_ns",
            return_value=([1], [9], [[11]], [[20260304, 1, 2]], [[123]]),
        ):
            rh.process_single_file("/tmp/abc.root", result)

        self.assertIn("9", result)
        self.assertEqual(result["9"]["file"], "abc.root")

        with patch.object(rh, "read_file_du_time_ns", side_effect=RuntimeError("x")):
            snapshot = dict(result)
            rh.process_single_file("/tmp/bad.root", result)
            self.assertEqual(result, snapshot)

    def test_mkdir_and_write_output(self) -> None:
        """Create directories and write YAML output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "a" / "b"

            rh.mkdir(str(out_dir))
            self.assertTrue(out_dir.exists())
            rh.mkdir(str(out_dir))
            self.assertTrue(out_dir.exists())

            data_dict = {
                "1": {
                    "run_number": 1,
                    "event_number": 1,
                    "time": {"11": [22]},
                }
            }
            file_path = rh.write_output(data_dict, str(out_dir), "out.yaml")
            self.assertEqual(Path(file_path), out_dir / "out.yaml")

            loaded = yaml.safe_load((out_dir / "out.yaml").read_text("utf-8"))
            self.assertEqual(loaded["1"]["event_number"], 1)

    def test_process_root_file_success(self) -> None:
        """Return True and write output when processing succeeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_base = Path(tmpdir) / "Reco_Dir"
            ok = rh.process_root_file(
                "Trigger_test.root",
                "2026/03/04",
                str(out_base),
            )

            out_yaml = (
                out_base
                / "2026/03/04"
                / "Trigger_test.yaml"
            )
            self.assertTrue(ok)
            self.assertTrue(out_yaml.exists())
            loaded = yaml.safe_load(out_yaml.read_text("utf-8"))
            self.assertEqual(loaded, {})

    def test_process_root_file_failure(self) -> None:
        """Return False when unexpected errors happen during processing."""
        with patch.object(rh, "mkdir", side_effect=RuntimeError("mkdir failed")):
            ok = rh.process_root_file("Trigger_test.root", "d", "/tmp/x")
        self.assertFalse(ok)

    def test_main_success_and_failure(self) -> None:
        """Return shell-style status code from process_root_file result."""
        with patch.object(
            rh,
            "parse_args",
            return_value=rh.argparse.Namespace(
                file_path="a.root",
                date="2026/03/04",
                out_dir_base="/tmp/out",
            ),
        ):
            with patch.object(rh, "process_root_file", return_value=True):
                self.assertEqual(rh.main(["prog"]), 0)
            with patch.object(rh, "process_root_file", return_value=False):
                self.assertEqual(rh.main(["prog"]), 2)

    def test_module_entrypoint_invokes_main(self) -> None:
        """Cover __main__ execution path and sys.argv forwarding."""
        module_path = Path(rh.__file__)
        init_globals = {"__name__": "__main__", "__file__": str(module_path)}

        with patch.object(rh, "main", return_value=0) as mocked_main:
            with patch.object(sys, "argv", ["read_header.py", "x.root"]):
                exec(module_path.read_text(encoding="utf-8"), init_globals)

        mocked_main.assert_not_called()

    def test_module_entrypoint_executes_main_line(self) -> None:
        """Execute module as script to cover the __main__ call line."""
        old_argv = sys.argv
        try:
            # Missing required file_path argument triggers argparse SystemExit.
            sys.argv = ["read_header.py"]
            with self.assertRaises(SystemExit):
                runpy.run_path(str(Path(rh.__file__)), run_name="__main__")
        finally:
            sys.argv = old_argv

    def test_loop_make_readheader_command_uses_new_script(self) -> None:
        """Build read-header command with renamed script path."""
        cmd = loop.make_readheader_command("a.root", "2026/03/04", "../Reco_Dir")
        self.assertEqual(cmd[1], "readroot/read_header.py")


if __name__ == "__main__":
    unittest.main()
