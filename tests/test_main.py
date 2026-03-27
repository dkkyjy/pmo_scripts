"""Unit tests for main.py cache read/write metadata behavior."""

from __future__ import annotations

import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import yaml

import main as main_module


class TestMainCacheSchema(unittest.TestCase):
    """Tests for cache schema consistency in main.py."""

    @staticmethod
    def _optimized_result() -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[int]]]:
        """Return a minimal optimized matching result."""
        return (
            {"1000_0": [1.0]},
            {"1000_0": [9.9]},
            {"1000_0": [101]},
        )

    @staticmethod
    def _pwm_result() -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, float], dict[str, float]]:
        """Return a minimal PWM fitting result."""
        return (
            {"1000_0": np.array([1.0, 0.0, 0.0])},
            {"1000_0": 1.23},
            {"1000_0": 2.34},
            {"1000_0": 0.12},
        )

    @staticmethod
    def _swm_result() -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Return a minimal SWM fitting result."""
        return (
            {"1000_0": np.array([0.1, 0.2, 0.3])},
            {"1000_0": 0.03},
        )

    def _prepare_inputs(self) -> tuple[Path, Path, Path]:
        """Create minimal matching and detector input files."""
        tmpdir = tempfile.TemporaryDirectory()
        root = Path(tmpdir.name)
        matching = root / "sample.yaml"
        det_pos = root / "det.txt"

        matching.write_text(
            yaml.safe_dump(
                {
                    "1000_0": {
                        "run_number": 11,
                        "event_number": 22,
                        "datetime": "2026-01-01T00:00:00",
                        "du_id": [101],
                        "file": "source.root",
                        "index": 7,
                        "time": {101: [1.0]},
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        det_pos.write_text("0 0 0\n", encoding="utf-8")
        self.addCleanup(tmpdir.cleanup)
        return root, matching, det_pos

    def _assert_cache_row_schema(
        self,
        cache_file: Path,
        event_key: str,
        expect_signal: bool,
    ) -> None:
        """Assert cache row contains required base fields and signal policy."""
        required = {
            "run_number",
            "event_number",
            "datetime",
            "gps_time",
            "du_id",
            "file",
            "index",
            "time",
        }
        payload = yaml.safe_load(cache_file.read_text(encoding="utf-8"))
        self.assertIn(event_key, payload, msg=f"Missing key in {cache_file}")
        row = payload[event_key]
        self.assertTrue(required.issubset(set(row.keys())))
        if expect_signal:
            self.assertIn("signal", row)
        else:
            self.assertNotIn("signal", row)

    @staticmethod
    def _write_cache(cache_file: Path, row: dict) -> None:
        """Write a one-row cache file."""
        cache_file.write_text(
            yaml.safe_dump({"1000_0": row}, sort_keys=False),
            encoding="utf-8",
        )

    @staticmethod
    def _write_stage_meta(stage: str, cache_file: Path, inputs: dict) -> None:
        """Write sidecar metadata for one stage cache."""
        payload = main_module._build_stage_cache_meta(stage, inputs)
        meta_file = main_module._meta_file_for(str(cache_file))
        main_module._write_cache_meta(meta_file, payload)

    @staticmethod
    def _matched_row(include_signal: bool) -> dict:
        """Return a matched-cache row."""
        row = {
            "run_number": 1,
            "event_number": 2,
            "datetime": "2026-01-01T00:00:00",
            "gps_time": 1704067200,
            "du_id": [101],
            "file": "source.root",
            "index": 0,
            "time": [1.0],
        }
        if include_signal:
            row["signal"] = [9.9]
        return row

    @staticmethod
    def _pwm_row(include_signal: bool) -> dict:
        """Return a PWM-cache row."""
        row = {
            "run_number": 1,
            "event_number": 2,
            "datetime": "2026-01-01T00:00:00",
            "gps_time": 1704067200,
            "du_id": [101],
            "file": "source.root",
            "index": 0,
            "time": [1.0],
            "chi_square": 0.12,
            "zenith": 1.23,
            "azimuth": 2.34,
            "x": 1.0,
            "y": 0.0,
            "z": 0.0,
        }
        if include_signal:
            row["signal"] = [9.9]
        return row

    @staticmethod
    def _swm_row(include_signal: bool) -> dict:
        """Return a SWM-cache row."""
        row = {
            "run_number": 1,
            "event_number": 2,
            "datetime": "2026-01-01T00:00:00",
            "gps_time": 1704067200,
            "du_id": [101],
            "file": "source.root",
            "index": 0,
            "time": [1.0],
            "chi_square": 0.03,
            "zenith": 1.23,
            "azimuth": 2.34,
            "x": 0.1,
            "y": 0.2,
            "z": 0.3,
        }
        if include_signal:
            row["signal"] = [9.9]
        return row

    def test_recompute_writes_matched_with_signal(self) -> None:
        """Recompute branch writes matched cache with signal when enabled."""
        root, matching, det_pos = self._prepare_inputs()

        with mock.patch("main.load_data_from_file", return_value={}), \
            mock.patch(
                "main.fe_mt.optimized_read_matching_times_graph",
                return_value=self._optimized_result(),
            ):
            result = main_module.main(
                str(matching),
                fig_name=None,
                with_signal=True,
                det_pos_file=str(det_pos),
                run_matching=True,
            )

        self.assertEqual(result, 0)

        cache_file = root / "sample_matched.yaml"
        self.assertTrue(cache_file.exists(), msg=f"Missing cache {cache_file}")
        self._assert_cache_row_schema(
            cache_file=cache_file,
            event_key="1000_0",
            expect_signal=True,
        )

    def test_recompute_writes_matched_without_signal(self) -> None:
        """Recompute branch omits signal in matched cache when disabled."""
        root, matching, det_pos = self._prepare_inputs()

        with mock.patch("main.load_data_from_file", return_value={}), \
            mock.patch(
                "main.fe_mt.optimized_read_matching_times_graph",
                return_value=self._optimized_result(),
            ):
            result = main_module.main(
                str(matching),
                fig_name=None,
                with_signal=False,
                det_pos_file=str(det_pos),
                run_matching=True,
            )

        self.assertEqual(result, 0)

        cache_file = root / "sample_matched.yaml"
        self.assertTrue(cache_file.exists(), msg=f"Missing cache {cache_file}")
        self._assert_cache_row_schema(
            cache_file=cache_file,
            event_key="1000_0",
            expect_signal=False,
        )

    def test_uses_existing_caches_without_recompute_calls(self) -> None:
        """If caches exist, recompute functions should not be called."""
        root, matching, det_pos = self._prepare_inputs()

        self._write_cache(root / "sample_matched.yaml", self._matched_row(False))
        self._write_cache(root / "sample_PWM.yaml", self._pwm_row(False))
        self._write_cache(root / "sample_SWM.yaml", self._swm_row(False))
        self._write_stage_meta(
            "matching",
            root / "sample_matched.yaml",
            {
                "matching_file": main_module._build_file_signature(str(matching)),
                "det_pos_file": main_module._build_file_signature(str(det_pos)),
                "with_signal": False,
            },
        )
        self._write_stage_meta(
            "pwm",
            root / "sample_PWM.yaml",
            {
                "matched_file": main_module._build_file_signature(
                    str(root / "sample_matched.yaml")
                ),
                "det_pos_file": main_module._build_file_signature(str(det_pos)),
                "with_signal": False,
            },
        )
        self._write_stage_meta(
            "swm",
            root / "sample_SWM.yaml",
            {
                "pwm_file": main_module._build_file_signature(
                    str(root / "sample_PWM.yaml")
                ),
                "det_pos_file": main_module._build_file_signature(str(det_pos)),
                "with_signal": False,
            },
        )

        with mock.patch("main.load_data_from_file", return_value={}), \
            mock.patch("main.fe_plot.plot_reconstructed_positions_PWM"), \
            mock.patch("main.fe_plot.plot_fitting_parameters_PWM"), \
            mock.patch("main.fe_plot.plot_reconstructed_positions_SWM"), \
            mock.patch("main.fe_plot.plot_fitting_parameters_SWM"), \
            mock.patch("main.fe_mt.optimized_read_matching_times_graph") as mt_mock, \
            mock.patch("main.fe_pwm.plane_wave_model") as pwm_mock, \
            mock.patch("main.fe_swm.spherical_wave_model") as swm_mock:
            result = main_module.main(
                str(matching),
                fig_name=None,
                with_signal=False,
                det_pos_file=str(det_pos),
            )

        self.assertEqual(result, 0)

        mt_mock.assert_not_called()
        pwm_mock.assert_not_called()
        swm_mock.assert_not_called()

    def test_default_runs_all_stages_and_writes_all_caches(self) -> None:
        """Default execution should run matching, PWM, and SWM in order."""
        root, matching, det_pos = self._prepare_inputs()

        with mock.patch("main.load_data_from_file", return_value={}), \
            mock.patch(
                "main.fe_mt.optimized_read_matching_times_graph",
                return_value=self._optimized_result(),
            ) as mt_mock, \
            mock.patch(
                "main.fe_pwm.plane_wave_model",
                return_value=self._pwm_result(),
            ) as pwm_mock, \
            mock.patch(
                "main.fe_swm.spherical_wave_model",
                return_value=self._swm_result(),
            ) as swm_mock, \
            mock.patch("main.fe_plot.plot_reconstructed_positions_PWM"), \
            mock.patch("main.fe_plot.plot_fitting_parameters_PWM"), \
            mock.patch("main.fe_plot.plot_reconstructed_positions_SWM"), \
            mock.patch("main.fe_plot.plot_fitting_parameters_SWM"):
            result = main_module.main(
                str(matching),
                fig_name=None,
                with_signal=True,
                det_pos_file=str(det_pos),
            )

        self.assertEqual(result, 0)
        mt_mock.assert_called_once()
        pwm_mock.assert_called_once()
        swm_mock.assert_called_once()
        self.assertTrue((root / "sample_matched.yaml").exists())
        self.assertTrue((root / "sample_PWM.yaml").exists())
        self.assertTrue((root / "sample_SWM.yaml").exists())

    def test_run_pwm_backfills_matching_without_swm(self) -> None:
        """Selecting only PWM should auto-run matching and skip SWM."""
        root, matching, det_pos = self._prepare_inputs()

        with mock.patch("main.load_data_from_file", return_value={}), \
            mock.patch(
                "main.fe_mt.optimized_read_matching_times_graph",
                return_value=self._optimized_result(),
            ) as mt_mock, \
            mock.patch(
                "main.fe_pwm.plane_wave_model",
                return_value=self._pwm_result(),
            ) as pwm_mock, \
            mock.patch("main.fe_swm.spherical_wave_model") as swm_mock, \
            mock.patch("main.fe_plot.plot_reconstructed_positions_PWM"), \
            mock.patch("main.fe_plot.plot_fitting_parameters_PWM"):
            result = main_module.main(
                str(matching),
                fig_name=None,
                with_signal=True,
                det_pos_file=str(det_pos),
                run_pwm=True,
            )

        self.assertEqual(result, 0)
        mt_mock.assert_called_once()
        pwm_mock.assert_called_once()
        swm_mock.assert_not_called()
        self.assertTrue((root / "sample_matched.yaml").exists())
        self.assertTrue((root / "sample_PWM.yaml").exists())
        self.assertFalse((root / "sample_SWM.yaml").exists())

    def test_run_swm_uses_existing_prerequisite_caches(self) -> None:
        """Selecting only SWM should reuse matched/PWM caches when available."""
        root, matching, det_pos = self._prepare_inputs()
        self._write_cache(root / "sample_matched.yaml", self._matched_row(True))
        self._write_cache(root / "sample_PWM.yaml", self._pwm_row(True))
        self._write_stage_meta(
            "matching",
            root / "sample_matched.yaml",
            {
                "matching_file": main_module._build_file_signature(str(matching)),
                "det_pos_file": main_module._build_file_signature(str(det_pos)),
                "with_signal": True,
            },
        )
        self._write_stage_meta(
            "pwm",
            root / "sample_PWM.yaml",
            {
                "matched_file": main_module._build_file_signature(
                    str(root / "sample_matched.yaml")
                ),
                "det_pos_file": main_module._build_file_signature(str(det_pos)),
                "with_signal": True,
            },
        )

        with mock.patch("main.load_data_from_file", return_value={}), \
            mock.patch("main.fe_mt.optimized_read_matching_times_graph") as mt_mock, \
            mock.patch("main.fe_pwm.plane_wave_model") as pwm_mock, \
            mock.patch(
                "main.fe_swm.spherical_wave_model",
                return_value=self._swm_result(),
            ) as swm_mock, \
            mock.patch("main.fe_plot.plot_reconstructed_positions_SWM"), \
            mock.patch("main.fe_plot.plot_fitting_parameters_SWM"):
            result = main_module.main(
                str(matching),
                fig_name=None,
                with_signal=True,
                det_pos_file=str(det_pos),
                run_swm=True,
            )

        self.assertEqual(result, 0)
        mt_mock.assert_not_called()
        pwm_mock.assert_not_called()
        swm_mock.assert_called_once()
        self.assertTrue((root / "sample_SWM.yaml").exists())

    def test_force_recompute_pwm_recomputes_dependency_chain(self) -> None:
        """force_recompute with PWM selected should recompute matching and PWM."""
        root, matching, det_pos = self._prepare_inputs()
        self._write_cache(root / "sample_matched.yaml", self._matched_row(False))
        self._write_cache(root / "sample_PWM.yaml", self._pwm_row(False))

        with mock.patch("main.load_data_from_file", return_value={}), \
            mock.patch(
                "main.fe_mt.optimized_read_matching_times_graph",
                return_value=self._optimized_result(),
            ) as mt_mock, \
            mock.patch(
                "main.fe_pwm.plane_wave_model",
                return_value=self._pwm_result(),
            ) as pwm_mock, \
            mock.patch("main.fe_swm.spherical_wave_model") as swm_mock, \
            mock.patch("main.fe_plot.plot_reconstructed_positions_PWM"), \
            mock.patch("main.fe_plot.plot_fitting_parameters_PWM"):
            result = main_module.main(
                str(matching),
                fig_name=None,
                with_signal=False,
                det_pos_file=str(det_pos),
                force_recompute=True,
                run_pwm=True,
            )

        self.assertEqual(result, 0)
        mt_mock.assert_called_once()
        pwm_mock.assert_called_once()
        swm_mock.assert_not_called()

    def test_matching_cache_signature_mismatch_recomputes(self) -> None:
        """Cache should be ignored when matching-stage meta signature mismatches."""
        root, matching, det_pos = self._prepare_inputs()
        self._write_cache(root / "sample_matched.yaml", self._matched_row(False))

        # Write intentionally stale meta: old with_signal value forces mismatch.
        self._write_stage_meta(
            "matching",
            root / "sample_matched.yaml",
            {
                "matching_file": main_module._build_file_signature(str(matching)),
                "det_pos_file": main_module._build_file_signature(str(det_pos)),
                "with_signal": False,
            },
        )

        with mock.patch("main.load_data_from_file", return_value={}), \
            mock.patch(
                "main.fe_mt.optimized_read_matching_times_graph",
                return_value=self._optimized_result(),
            ) as mt_mock:
            result = main_module.main(
                str(matching),
                fig_name=None,
                with_signal=True,
                det_pos_file=str(det_pos),
                run_matching=True,
            )

        self.assertEqual(result, 0)
        mt_mock.assert_called_once()

    def test_invalid_pwm_cache_payload_falls_back_to_recompute(self) -> None:
        """Missing required PWM cache fields should trigger recompute fallback."""
        root, matching, det_pos = self._prepare_inputs()
        self._write_cache(root / "sample_matched.yaml", self._matched_row(False))

        invalid_pwm_row = self._pwm_row(False)
        invalid_pwm_row.pop("x")
        self._write_cache(root / "sample_PWM.yaml", invalid_pwm_row)

        self._write_stage_meta(
            "matching",
            root / "sample_matched.yaml",
            {
                "matching_file": main_module._build_file_signature(str(matching)),
                "det_pos_file": main_module._build_file_signature(str(det_pos)),
                "with_signal": False,
            },
        )
        self._write_stage_meta(
            "pwm",
            root / "sample_PWM.yaml",
            {
                "matched_file": main_module._build_file_signature(
                    str(root / "sample_matched.yaml")
                ),
                "det_pos_file": main_module._build_file_signature(str(det_pos)),
                "with_signal": False,
            },
        )

        with mock.patch("main.load_data_from_file", return_value={}), \
            mock.patch("main.fe_mt.optimized_read_matching_times_graph") as mt_mock, \
            mock.patch(
                "main.fe_pwm.plane_wave_model",
                return_value=self._pwm_result(),
            ) as pwm_mock, \
            mock.patch("main.fe_plot.plot_reconstructed_positions_PWM"), \
            mock.patch("main.fe_plot.plot_fitting_parameters_PWM"):
            result = main_module.main(
                str(matching),
                fig_name=None,
                with_signal=False,
                det_pos_file=str(det_pos),
                run_pwm=True,
            )

        self.assertEqual(result, 0)
        mt_mock.assert_not_called()
        pwm_mock.assert_called_once()

    def test_det_pos_missing_returns_2_without_stage_execution(self) -> None:
        """Missing det-pos file should return stable error code 2."""
        _root, matching, det_pos = self._prepare_inputs()
        missing_det_pos = str(det_pos) + ".missing"

        with mock.patch("main.load_data_from_file") as load_mock, \
            mock.patch("main.fe_mt.optimized_read_matching_times_graph") as mt_mock:
            result = main_module.main(
                str(matching),
                fig_name=None,
                with_signal=False,
                det_pos_file=missing_det_pos,
                run_matching=True,
            )

        self.assertEqual(result, 2)
        load_mock.assert_not_called()
        mt_mock.assert_not_called()

    def test_det_pos_load_value_error_returns_2(self) -> None:
        """det-pos parse/load failure should return stable error code 2."""
        _root, matching, det_pos = self._prepare_inputs()

        with mock.patch("main.load_data_from_file", side_effect=ValueError("bad format")), \
            mock.patch("main.fe_mt.optimized_read_matching_times_graph") as mt_mock:
            result = main_module.main(
                str(matching),
                fig_name=None,
                with_signal=False,
                det_pos_file=str(det_pos),
                run_matching=True,
            )

        self.assertEqual(result, 2)
        mt_mock.assert_not_called()


def _prepare_stage_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal matching and detector files for direct stage tests."""
    matching = tmp_path / "sample.yaml"
    det_pos = tmp_path / "det.txt"
    matching.write_text(
        yaml.safe_dump(
            {
                "1000_0": {
                    "run_number": 1,
                    "event_number": 2,
                    "datetime": "2026-01-01T00:00:00",
                    "gps_time": 1704067200,
                    "du_id": [101],
                    "file": "source.root",
                    "index": 0,
                    "time": {101: [1.0]},
                },
                "skip": "not-a-dict",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    det_pos.write_text("0 0 0\n", encoding="utf-8")
    return matching, det_pos


def _seed_matching_state(with_signal: bool = False) -> dict:
    """Return a minimal state that looks like matching-stage output."""
    state = main_module._new_stage_state()
    state["times"]["1000_0"] = [1.0]
    state["signals"]["1000_0"] = [9.9] if with_signal else None
    state["du_ids"]["1000_0"] = [101]
    state["run_numbers"]["1000_0"] = 1
    state["event_numbers"]["1000_0"] = 2
    state["gps_times"]["1000_0"] = 1704067200
    state["files"]["1000_0"] = "source.root"
    state["index"]["1000_0"] = 0
    state["datetimes"]["1000_0"] = "2026-01-01T00:00:00"
    return state


def _seed_pwm_state(with_signal: bool = False) -> dict:
    """Return a minimal state that looks like PWM-stage input/output."""
    state = _seed_matching_state(with_signal=with_signal)
    state["azimuths"]["1000_0"] = 2.34
    state["zeniths"]["1000_0"] = 1.23
    state["directions"]["1000_0"] = np.array([1.0, 0.0, 0.0])
    state["chi_squares"]["1000_0"] = 0.12
    return state


def test_load_event_metadata_map_returns_empty_for_missing_file(tmp_path: Path) -> None:
    """Missing matching files should produce an empty metadata map."""
    assert main_module._load_event_metadata_map(str(tmp_path / "missing.yaml")) == {}


def test_load_event_metadata_map_returns_empty_for_non_dict_payload(tmp_path: Path) -> None:
    """Non-dict YAML payloads should be rejected."""
    matching = tmp_path / "sample.yaml"
    matching.write_text(yaml.safe_dump([1, 2, 3]), encoding="utf-8")

    assert main_module._load_event_metadata_map(str(matching)) == {}


def test_load_event_metadata_map_skips_non_dict_rows(tmp_path: Path) -> None:
    """Non-dict rows should be ignored while valid rows are preserved."""
    matching, _det_pos = _prepare_stage_inputs(tmp_path)

    metadata = main_module._load_event_metadata_map(str(matching))

    assert metadata == {
        "1000_0": {
            "run_number": 1,
            "event_number": 2,
            "datetime": "2026-01-01T00:00:00",
            "gps_time": 1704067200,
            "file": "source.root",
            "index": 0,
        }
    }


def test_parse_args_supports_all_flags() -> None:
    """CLI parser should accept all supported stage and cache flags."""
    args = main_module.parse_args(
        [
            "main.py",
            "input.yaml",
            "--fig_name",
            "fig",
            "--det-pos",
            "det.txt",
            "--with-signal",
            "--force-recompute",
            "--run-matching",
            "--run-pwm",
            "--run-swm",
        ]
    )

    assert args.matching_file == "input.yaml"
    assert args.fig_name == "fig"
    assert args.det_pos == "det.txt"
    assert args.with_signal is True
    assert args.force_recompute is True
    assert args.run_matching is True
    assert args.run_pwm is True
    assert args.run_swm is True


def test_build_file_signature_handles_missing_file(tmp_path: Path) -> None:
    """Missing files should report a non-existing signature."""
    signature = main_module._build_file_signature(str(tmp_path / "missing.txt"))

    assert signature["exists"] is False
    assert signature["path"].endswith("missing.txt")


def test_read_cache_meta_handles_missing_invalid_type_and_json_error(
    tmp_path: Path,
) -> None:
    """Cache meta reader should return None for missing, invalid, and unreadable files."""
    missing = tmp_path / "missing.meta.json"
    assert main_module._read_cache_meta(str(missing)) is None

    invalid_type = tmp_path / "invalid-type.meta.json"
    invalid_type.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert main_module._read_cache_meta(str(invalid_type)) is None

    unreadable = tmp_path / "unreadable.meta.json"
    unreadable.write_text("{bad json", encoding="utf-8")
    assert main_module._read_cache_meta(str(unreadable)) is None


def test_cache_meta_is_valid_reports_missing_and_mismatch(tmp_path: Path) -> None:
    """Meta validation should distinguish missing and mismatched signatures."""
    meta_file = tmp_path / "cache.meta.json"
    expected = {"schema_version": 1}

    assert main_module._cache_meta_is_valid(str(meta_file), expected) == (
        False,
        "missing-or-unreadable",
    )

    meta_file.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    assert main_module._cache_meta_is_valid(str(meta_file), expected) == (
        False,
        "signature-mismatch",
    )


def test_required_fields_for_stage_raises_on_unknown_stage() -> None:
    """Unknown stage names should be rejected explicitly."""
    with pytest.raises(ValueError):
        main_module._required_fields_for_stage("unknown", False)


def test_validate_stage_cache_payload_rejects_bad_top_level_and_row() -> None:
    """Cache payload validator should reject non-dict payloads and rows."""
    assert main_module._validate_stage_cache_payload("matching", [], False) == (
        False,
        "top-level payload is not a dict",
    )
    assert main_module._validate_stage_cache_payload(
        "matching",
        {"1000_0": []},
        False,
    ) == (False, "row is not a dict for event 1000_0")


def test_plot_pwm_if_needed_success_and_exception() -> None:
    """PWM plotting should call plotters and swallow plotting errors."""
    state = _seed_pwm_state()
    state["times"]["1000_1"] = [2.0]
    state["directions"]["1000_1"] = np.array([0.0, 1.0, 0.0])
    state["chi_squares"]["1000_1"] = 0.34
    state["datetimes"]["1000_1"] = "2026-01-01T00:00:01"

    with mock.patch("main.fe_plot.plot_reconstructed_positions_PWM") as pos_mock, \
        mock.patch("main.fe_plot.plot_fitting_parameters_PWM") as fit_mock:
        main_module._plot_pwm_if_needed(state, "fig")

    pos_mock.assert_called_once()
    fit_mock.assert_called_once()

    with mock.patch(
        "main.fe_plot.plot_reconstructed_positions_PWM",
        side_effect=RuntimeError("plot failed"),
    ), mock.patch("main.fe_plot.plot_fitting_parameters_PWM"):
        main_module._plot_pwm_if_needed(state, "fig")


def test_plot_swm_if_needed_success_and_exception() -> None:
    """SWM plotting should call plotters and swallow plotting errors."""
    state = _seed_pwm_state()
    state["times"]["1000_1"] = [2.0]
    state["directions"]["1000_1"] = np.array([0.0, 1.0, 0.0])
    state["chi_squares"]["1000_1"] = 0.34
    state["datetimes"]["1000_1"] = "2026-01-01T00:00:01"

    with mock.patch("main.fe_plot.plot_reconstructed_positions_SWM") as pos_mock, \
        mock.patch("main.fe_plot.plot_fitting_parameters_SWM") as fit_mock:
        main_module._plot_swm_if_needed(state, "fig")

    pos_mock.assert_called_once()
    fit_mock.assert_called_once()

    with mock.patch(
        "main.fe_plot.plot_reconstructed_positions_SWM",
        side_effect=RuntimeError("plot failed"),
    ), mock.patch("main.fe_plot.plot_fitting_parameters_SWM"):
        main_module._plot_swm_if_needed(state, "fig")


def test_run_matching_stage_returns_existing_state_without_work(tmp_path: Path) -> None:
    """Prepopulated matching state should short-circuit without cache access."""
    matching, det_pos = _prepare_stage_inputs(tmp_path)
    state = _seed_matching_state()

    returned_state, matching_computed = main_module.run_matching_stage(
        str(matching),
        {},
        str(det_pos),
        False,
        False,
        state,
    )

    assert returned_state is state
    assert matching_computed is False


def test_run_matching_stage_handles_unreadable_and_invalid_cache_payloads(
    tmp_path: Path,
) -> None:
    """Unreadable or invalid matched caches should fall back to recompute."""
    matching, det_pos = _prepare_stage_inputs(tmp_path)
    matched_file = tmp_path / "sample_matched.yaml"
    matched_file.write_text("cached", encoding="utf-8")

    expected_meta = main_module._build_stage_cache_meta(
        "matching",
        {
            "matching_file": main_module._build_file_signature(str(matching)),
            "det_pos_file": main_module._build_file_signature(str(det_pos)),
            "with_signal": False,
        },
    )
    main_module._write_cache_meta(
        main_module._meta_file_for(str(matched_file)),
        expected_meta,
    )

    def read_yaml_side_effect(file_path: str) -> dict:
        if "_matched.yaml" in file_path:
            raise ValueError("broken cache")
        # Return valid raw input data for the matching_file call
        return {
            "1000_0": {
                "run_number": 1,
                "event_number": 2,
                "datetime": "2026-01-01T00:00:00",
                "du_id": [101],
                "file": "source.root",
                "index": 0,
                "time": {101: [1.0]},
            }
        }

    with mock.patch(
        "main._read_yaml_dict",
        side_effect=read_yaml_side_effect,
    ), mock.patch(
        "main.fe_mt.optimized_read_matching_times_graph",
        return_value=TestMainCacheSchema._optimized_result(),
    ) as mt_mock:
        state, matching_computed = main_module.run_matching_stage(
            str(matching),
            main_module._load_event_metadata_map(str(matching)),
            str(det_pos),
            False,
            False,
            main_module._new_stage_state(),
        )

    assert matching_computed is True
    mt_mock.assert_called_once()
    assert state["times"]["1000_0"] == [1.0]

    invalid_payload = {"1000_0": {"run_number": 1}}
    matched_file.write_text(yaml.safe_dump(invalid_payload), encoding="utf-8")

    with mock.patch(
        "main.fe_mt.optimized_read_matching_times_graph",
        return_value=TestMainCacheSchema._optimized_result(),
    ) as mt_mock:
        _state, matching_computed = main_module.run_matching_stage(
            str(matching),
            main_module._load_event_metadata_map(str(matching)),
            str(det_pos),
            False,
            False,
            main_module._new_stage_state(),
        )

    assert matching_computed is True
    mt_mock.assert_called_once()


def test_run_matching_stage_returns_early_when_no_events(tmp_path: Path) -> None:
    """Matching stage should stop the pipeline cleanly when filtering removes all events."""
    matching, det_pos = _prepare_stage_inputs(tmp_path)

    with mock.patch(
        "main.fe_mt.optimized_read_matching_times_graph",
        return_value=({}, {}, {}),
    ):
        state, matching_computed = main_module.run_matching_stage(
            str(matching),
            main_module._load_event_metadata_map(str(matching)),
            str(det_pos),
            False,
            False,
            main_module._new_stage_state(),
        )

    assert state["times"] == {}
    assert matching_computed is True


def test_run_pwm_stage_recomputes_for_signature_mismatch_and_unreadable_cache(
    tmp_path: Path,
) -> None:
    """PWM stage should recompute when cache meta mismatches or cache load fails."""
    matching, det_pos = _prepare_stage_inputs(tmp_path)
    matched_file = tmp_path / "sample_matched.yaml"
    matched_file.write_text(yaml.safe_dump({"1000_0": TestMainCacheSchema._matched_row(False)}), encoding="utf-8")
    pwm_file = tmp_path / "sample_PWM.yaml"
    pwm_file.write_text("cached", encoding="utf-8")

    mismatched_meta = main_module._build_stage_cache_meta(
        "pwm",
        {
            "matched_file": main_module._build_file_signature(str(matched_file)),
            "det_pos_file": main_module._build_file_signature(str(det_pos)),
            "with_signal": True,
        },
    )
    main_module._write_cache_meta(main_module._meta_file_for(str(pwm_file)), mismatched_meta)

    with mock.patch(
        "main.fe_pwm.plane_wave_model",
        return_value=TestMainCacheSchema._pwm_result(),
    ) as pwm_mock, mock.patch("main.fe_plot.plot_reconstructed_positions_PWM"), \
        mock.patch("main.fe_plot.plot_fitting_parameters_PWM"):
        _state, pwm_computed = main_module.run_pwm_stage(
            str(matching),
            {},
            str(det_pos),
            False,
            False,
            _seed_matching_state(),
            str(tmp_path / "fig"),
            False,
        )

    assert pwm_computed is True
    pwm_mock.assert_called_once()

    expected_meta = main_module._build_stage_cache_meta(
        "pwm",
        {
            "matched_file": main_module._build_file_signature(str(matched_file)),
            "det_pos_file": main_module._build_file_signature(str(det_pos)),
            "with_signal": False,
        },
    )
    main_module._write_cache_meta(main_module._meta_file_for(str(pwm_file)), expected_meta)

    with mock.patch("main._read_yaml_dict", side_effect=ValueError("broken cache")), \
        mock.patch(
            "main.fe_pwm.plane_wave_model",
            return_value=TestMainCacheSchema._pwm_result(),
        ) as pwm_mock, mock.patch("main.fe_plot.plot_reconstructed_positions_PWM"), \
        mock.patch("main.fe_plot.plot_fitting_parameters_PWM"):
        _state, pwm_computed = main_module.run_pwm_stage(
            str(matching),
            {},
            str(det_pos),
            False,
            False,
            _seed_matching_state(),
            str(tmp_path / "fig"),
            False,
        )

    assert pwm_computed is True
    pwm_mock.assert_called_once()


def test_run_swm_stage_recomputes_for_all_cache_fallback_paths(tmp_path: Path) -> None:
    """SWM stage should recompute for signature mismatch, unreadable cache, invalid payload, and force-recompute."""
    matching, det_pos = _prepare_stage_inputs(tmp_path)
    pwm_file = tmp_path / "sample_PWM.yaml"
    pwm_file.write_text(yaml.safe_dump({"1000_0": TestMainCacheSchema._pwm_row(False)}), encoding="utf-8")
    swm_file = tmp_path / "sample_SWM.yaml"
    swm_file.write_text("cached", encoding="utf-8")

    mismatched_meta = main_module._build_stage_cache_meta(
        "swm",
        {
            "pwm_file": main_module._build_file_signature(str(pwm_file)),
            "det_pos_file": main_module._build_file_signature(str(det_pos)),
            "with_signal": True,
        },
    )
    main_module._write_cache_meta(main_module._meta_file_for(str(swm_file)), mismatched_meta)

    with mock.patch(
        "main.fe_swm.spherical_wave_model",
        return_value=TestMainCacheSchema._swm_result(),
    ) as swm_mock, mock.patch("main.fe_plot.plot_reconstructed_positions_SWM"), \
        mock.patch("main.fe_plot.plot_fitting_parameters_SWM"):
        state = _seed_pwm_state()
        returned_state = main_module.run_swm_stage(
            str(matching),
            {},
            str(det_pos),
            False,
            False,
            state,
            str(tmp_path / "fig"),
            False,
        )

    assert returned_state["chi_squares"]["1000_0"] == 0.03
    swm_mock.assert_called_once()

    expected_meta = main_module._build_stage_cache_meta(
        "swm",
        {
            "pwm_file": main_module._build_file_signature(str(pwm_file)),
            "det_pos_file": main_module._build_file_signature(str(det_pos)),
            "with_signal": False,
        },
    )
    main_module._write_cache_meta(main_module._meta_file_for(str(swm_file)), expected_meta)

    with mock.patch("main._read_yaml_dict", side_effect=ValueError("broken cache")), \
        mock.patch(
            "main.fe_swm.spherical_wave_model",
            return_value=TestMainCacheSchema._swm_result(),
        ) as swm_mock, mock.patch("main.fe_plot.plot_reconstructed_positions_SWM"), \
        mock.patch("main.fe_plot.plot_fitting_parameters_SWM"):
        main_module.run_swm_stage(
            str(matching),
            {},
            str(det_pos),
            False,
            False,
            _seed_pwm_state(),
            str(tmp_path / "fig"),
            False,
        )

    swm_mock.assert_called_once()

    invalid_payload = {"1000_0": {"run_number": 1}}
    swm_file.write_text(yaml.safe_dump(invalid_payload), encoding="utf-8")

    with mock.patch(
        "main.fe_swm.spherical_wave_model",
        return_value=TestMainCacheSchema._swm_result(),
    ) as swm_mock, mock.patch("main.fe_plot.plot_reconstructed_positions_SWM"), \
        mock.patch("main.fe_plot.plot_fitting_parameters_SWM"):
        main_module.run_swm_stage(
            str(matching),
            {},
            str(det_pos),
            False,
            False,
            _seed_pwm_state(),
            str(tmp_path / "fig"),
            False,
        )

    swm_mock.assert_called_once()

    with mock.patch(
        "main.fe_swm.spherical_wave_model",
        return_value=TestMainCacheSchema._swm_result(),
    ) as swm_mock, mock.patch("main.fe_plot.plot_reconstructed_positions_SWM"), \
        mock.patch("main.fe_plot.plot_fitting_parameters_SWM"): 
        main_module.run_swm_stage(
            str(matching),
            {},
            str(det_pos),
            False,
            True,
            _seed_pwm_state(),
            str(tmp_path / "fig"),
            False,
        )

    swm_mock.assert_called_once()


def test_main_returns_zero_when_matching_stage_leaves_no_events(tmp_path: Path) -> None:
    """main should return 0 when matching stage produces no surviving events."""
    matching, det_pos = _prepare_stage_inputs(tmp_path)

    with mock.patch("main.load_data_from_file", return_value={}), \
        mock.patch(
            "main.run_matching_stage",
            return_value=(main_module._new_stage_state(), False),
        ):
        result = main_module.main(
            str(matching),
            fig_name=None,
            with_signal=False,
            det_pos_file=str(det_pos),
            run_matching=True,
        )

    assert result == 0


def test_main_module_main_guard_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Executing main as __main__ should raise SystemExit with main() result."""
    matching, _det_pos = _prepare_stage_inputs(tmp_path)
    missing_det_pos = tmp_path / "missing.txt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            str(matching),
            "--det-pos",
            str(missing_det_pos),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("main", run_name="__main__")

    assert exc.value.code == 2


if __name__ == "__main__":
    unittest.main()
