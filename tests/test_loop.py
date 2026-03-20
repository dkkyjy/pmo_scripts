"""Tests for loop.py."""

from __future__ import annotations

import builtins
import datetime as dt
import importlib.util
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

import loop


def test_parse_datetime_supports_multiple_formats() -> None:
    """parse_datetime should parse all supported input formats."""
    assert loop.parse_datetime("2025-11-27T15:30:00") == dt.datetime(
        2025, 11, 27, 15, 30, 0
    )
    assert loop.parse_datetime("20251127153000") == dt.datetime(
        2025, 11, 27, 15, 30, 0
    )
    assert loop.parse_datetime("2025-11-271530") == dt.datetime(
        2025, 11, 27, 15, 30, 0
    )
    assert loop.parse_datetime("2025/11/271530") == dt.datetime(
        2025, 11, 27, 15, 30, 0
    )


def test_parse_datetime_raises_on_invalid() -> None:
    """Invalid datetime string should raise ValueError."""
    with pytest.raises(ValueError):
        loop.parse_datetime("not-a-datetime")


def test_parse_datetime_pads_short_numeric_value() -> None:
    """Short numeric datetime should be right-padded before parsing."""
    assert loop.parse_datetime("20251127153") == dt.datetime(
        2025, 11, 27, 15, 30, 0
    )


def test_times_between_is_inclusive_and_midnight() -> None:
    """times_between should include both ends and normalize to midnight."""
    start = dt.datetime(2025, 1, 1, 12, 34, 56)
    end = dt.datetime(2025, 1, 3, 1, 0, 0)

    values = list(loop.times_between(start, end))
    assert values == [
        dt.datetime(2025, 1, 1, 0, 0, 0),
        dt.datetime(2025, 1, 2, 0, 0, 0),
        dt.datetime(2025, 1, 3, 0, 0, 0),
    ]


def test_get_file_bounds_for_date_middle_day() -> None:
    """Middle-day bounds should be clipped to current day 00:00~24:00."""
    start = dt.datetime(2025, 1, 1, 10, 0, 0)
    end = dt.datetime(2025, 1, 3, 20, 0, 0)
    current = dt.datetime(2025, 1, 2, 0, 0, 0)

    s, e = loop.get_file_bounds_for_date(start, end, current)
    assert s == 20250102000000
    assert e == 20250102240000


def test_get_file_list_filters_trigger_root_and_range(tmp_path: Path) -> None:
    """get_file_list should return only matching Trigger root files in range."""
    date_dir = tmp_path / "2025" / "11" / "27"
    date_dir.mkdir(parents=True)

    # matching
    (date_dir / "Trigger_20251127120000_x.root").write_text("")
    (date_dir / "Trigger_20251127125959_x.root").write_text("")
    # non-matching
    (date_dir / "Trigger_20251127130000_x.txt").write_text("")
    (date_dir / "Trigger_20251127121500_x.root.bak").write_text("")
    (date_dir / "Noise_20251127120000_x.root").write_text("")
    (date_dir / "Trigger_bad_x.root").write_text("")

    result = loop.get_file_list(
        20251127120000, 20251127125959, "2025/11/27", str(tmp_path)
    )
    names = [Path(p).name for p in result]
    assert names == [
        "Trigger_20251127120000_x.root",
        "Trigger_20251127125959_x.root",
    ]


def test_get_file_list_returns_empty_when_directory_missing(tmp_path: Path) -> None:
    """Missing date directory should return an empty list."""
    assert loop.get_file_list(1, 2, "2025/11/27", str(tmp_path)) == []


def test_get_file_list_skips_names_without_second_segment(tmp_path: Path) -> None:
    """Malformed Trigger file names without a second segment should be ignored."""
    date_dir = tmp_path / "2025" / "11" / "27"
    date_dir.mkdir(parents=True)
    (date_dir / "Trigger.root").write_text("")

    assert loop.get_file_list(0, 99999999999999, "2025/11/27", str(tmp_path)) == []


def test_make_command_helpers() -> None:
    """Command builders should generate expected argv lists."""
    rh = loop.make_readheader_command("/a/b/Trigger_x.root", "2025/11/27", "Reco")
    assert rh[:2] == [loop.sys.executable, "readroot/read_header.py"]
    assert "--date" in rh and "--out_dir_base" in rh

    rt = loop.make_readtrace_command(
        "/a/b/Trigger_x.root", "2025/11/27", "Reco", left=10, right=20
    )
    assert rt[:2] == [loop.sys.executable, "readroot/read_trace.py"]
    assert "10" in rt and "20" in rt

    m0 = loop.make_main_command(
        "/a/b/Trigger_x.root", "2025/11/27", "Reco", with_signal=False, channel="X"
    )
    assert m0[:2] == [loop.sys.executable, "main.py"]
    assert "--with-signal" not in m0
    assert m0[-1] == "Trigger_x"

    m1 = loop.make_main_command(
        "/a/b/Trigger_x.root",
        "2025/11/27",
        "Reco",
        with_signal=True,
        channel="Y",
        run_matching=True,
        run_pwm=True,
    )
    assert "--with-signal" in m1
    assert "--run-matching" in m1
    assert "--run-pwm" in m1
    assert "--run-swm" not in m1


def test_make_main_command_includes_run_swm_flag() -> None:
    """run_swm should be forwarded to main.py when requested."""
    command = loop.make_main_command(
        "/a/b/Trigger_x.root",
        "2025/11/27",
        "Reco",
        run_swm=True,
    )

    assert "--run-swm" in command


def test_make_tasks_for_file_paths() -> None:
    """Task list should contain one tuple per input file."""
    args = SimpleNamespace(
        out_dir_base="Reco",
        run=True,
        skip_read=False,
        only_read=True,
        with_signal=False,
        left=0,
        right=512,
        channel="X",
        run_matching=True,
        run_pwm=False,
        run_swm=True,
    )
    tasks = loop.make_tasks_for_file_paths(["a.root", "b.root"], "2025/11/27", args)
    assert len(tasks) == 2
    assert tasks[0].file_path == "a.root"
    assert tasks[0].date == "2025/11/27"
    assert tasks[0].out_dir_base == "Reco"
    assert tasks[0].run_matching is True
    assert tasks[0].run_pwm is False
    assert tasks[0].run_swm is True


def test_build_tasks_uses_date_iterator_and_file_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_tasks should aggregate tasks from all dates."""
    d1 = dt.datetime(2025, 11, 27)
    d2 = dt.datetime(2025, 11, 28)

    monkeypatch.setattr(loop, "times_between", lambda _s, _e: [d1, d2])
    monkeypatch.setattr(
        loop,
        "get_file_list",
        lambda _fs, _fe, date, _bp: [f"/tmp/{date}/Trigger_a.root"],
    )

    args = SimpleNamespace(
        out_dir_base="Reco",
        run=False,
        skip_read=False,
        only_read=False,
        with_signal=False,
        left=0,
        right=512,
        channel="X",
        run_matching=False,
        run_pwm=False,
        run_swm=False,
        base_path="/base",
    )

    tasks = loop.build_tasks(d1, d2, args)
    assert len(tasks) == 2
    assert tasks[0].file_path.endswith("Trigger_a.root")
    assert tasks[1].date == "2025/11/28"


def test_execute_tasks_sequential(monkeypatch: pytest.MonkeyPatch) -> None:
    """execute_tasks(jobs<=1) should call process_date sequentially."""
    monkeypatch.setattr(loop, "process_date", lambda task: (task.file_path, 0))
    tasks = [
        loop.TaskSpec("a", "d", "o", False, False, False, False, 0, 512, "X", False, False, False),
        loop.TaskSpec("b", "d", "o", False, False, False, False, 0, 512, "X", False, False, False),
    ]
    res = loop.execute_tasks(tasks, jobs=1)
    assert res == [("a", 0), ("b", 0)]


def test_execute_tasks_parallel_uses_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """execute_tasks(jobs>1) should dispatch through multiprocessing.Pool."""
    seen: dict[str, object] = {}

    class FakePool:
        def __init__(self, processes: int) -> None:
            seen["processes"] = processes

        def __enter__(self) -> "FakePool":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            return None

        def map(self, func, tasks):
            seen["mapped"] = True
            return [func(task) for task in tasks]

    monkeypatch.setattr(loop.multiprocessing, "Pool", FakePool)
    monkeypatch.setattr(loop, "process_date", lambda task: (task.file_path, 0))

    tasks = [
        loop.TaskSpec("a", "d", "o", False, False, False, False, 0, 512, "X", False, False, False),
        loop.TaskSpec("b", "d", "o", False, False, False, False, 0, 512, "X", False, False, False),
    ]

    assert loop.execute_tasks(tasks, jobs=3) == [("a", 0), ("b", 0)]
    assert seen == {"processes": 3, "mapped": True}


def test_merge_images_to_pdf_returns_false_when_pil_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """merge_images_to_pdf should short-circuit when PIL is unavailable."""
    monkeypatch.setattr(loop, "PIL_AVAILABLE", False)
    assert loop.merge_images_to_pdf("x", False, ".") is False


def test_merge_images_to_pdf_uses_search_dir_without_fallback_on_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When search_dir has matches, cwd fallback should not be used."""
    calls: list[str] = []

    def fake_glob(pattern: str) -> list[str]:
        calls.append(pattern)
        return ["dummy.png"] if len(calls) == 1 else []

    monkeypatch.setattr(loop, "PIL_AVAILABLE", True)
    monkeypatch.setattr(loop.glob, "glob", fake_glob)
    monkeypatch.setattr(
        loop,
        "Image",
        SimpleNamespace(open=lambda _f: (_ for _ in ()).throw(OSError("bad image"))),
    )

    result = loop.merge_images_to_pdf("out/Trigger_x", False, "Reco")

    assert result is False
    assert calls == ["Reco/out/Trigger_x*.png"]


def test_merge_images_to_pdf_falls_back_to_cwd_when_search_dir_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When search_dir has no matches, function should retry with cwd pattern."""
    calls: list[str] = []

    def fake_glob(pattern: str) -> list[str]:
        calls.append(pattern)
        return []

    monkeypatch.setattr(loop, "PIL_AVAILABLE", True)
    monkeypatch.setattr(loop.glob, "glob", fake_glob)

    result = loop.merge_images_to_pdf("out/Trigger_x", False, "Reco")

    assert result is False
    assert calls == ["Reco/out/Trigger_x*.png", "out/Trigger_x*.png"]


def test_merge_images_to_pdf_without_search_dir_writes_pdf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When PNGs exist and save succeeds, PDF generation should return True."""
    saved: dict[str, object] = {}

    class FakeImage:
        def __init__(self, name: str) -> None:
            self.name = name

        def convert(self, mode: str) -> "FakeImage":
            assert mode == "RGB"
            return self

        def save(self, path: str, fmt: str, save_all: bool, append_images):
            saved["path"] = path
            saved["fmt"] = fmt
            saved["save_all"] = save_all
            saved["append_len"] = len(append_images)

    fake_images = {
        "a.png": FakeImage("a"),
        "b.png": FakeImage("b"),
    }

    monkeypatch.setattr(loop, "PIL_AVAILABLE", True)
    monkeypatch.setattr(loop.glob, "glob", lambda _pattern: ["a.png", "b.png"])
    monkeypatch.setattr(
        loop,
        "Image",
        SimpleNamespace(open=lambda file_name: fake_images[file_name]),
    )

    pdf_basename = tmp_path / "nested" / "Trigger_x"
    assert loop.merge_images_to_pdf(str(pdf_basename), False, None) is True
    assert saved == {
        "path": str(pdf_basename) + ".pdf",
        "fmt": "PDF",
        "save_all": True,
        "append_len": 1,
    }


def test_merge_images_to_pdf_returns_false_when_save_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """PDF save failures should be logged and converted to False."""

    class FakeImage:
        def convert(self, _mode: str) -> "FakeImage":
            return self

        def save(self, *_args, **_kwargs) -> None:
            raise OSError("save failed")

    monkeypatch.setattr(loop, "PIL_AVAILABLE", True)
    monkeypatch.setattr(loop.glob, "glob", lambda _pattern: ["a.png"])
    monkeypatch.setattr(
        loop,
        "Image",
        SimpleNamespace(open=lambda _file_name: FakeImage()),
    )

    assert loop.merge_images_to_pdf(str(tmp_path / "Trigger_x"), False, None) is False


def test_loop_module_sets_pil_unavailable_when_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Import failure for PIL should set PIL_AVAILABLE to False at module load."""
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # noqa: ANN001
        if name == "PIL":
            raise ImportError("missing pillow")
        return original_import(name, *args, **kwargs)

    module_name = "loop_no_pil"
    spec = importlib.util.spec_from_file_location(module_name, loop.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setitem(sys.modules, module_name, module)

    spec.loader.exec_module(module)

    assert module.PIL_AVAILABLE is False


def test_process_date_skip_and_only_read_no_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """skip_read + only_read should avoid subprocess execution."""
    called = {"run": 0}

    def fake_run(*_args, **_kwargs):
        called["run"] += 1
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(loop.subprocess, "run", fake_run)

    task = loop.TaskSpec("/tmp/a.root", "2025/11/27", "Reco", False, True, True, False, 0, 512, "X", False, False, False)
    result = loop.process_date(task)

    assert result == ("/tmp/a.root", 0)
    assert called["run"] == 0


def test_process_date_with_signal_readtrace_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """with_signal read stage non-zero should return that code."""
    called = []

    def fake_run(cmd, check=False):  # noqa: ARG001
        called.append(cmd[1])
        if "readroot/read_trace.py" in cmd:
            return SimpleNamespace(returncode=9)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(loop.subprocess, "run", fake_run)

    task = loop.TaskSpec("/tmp/a.root", "2025/11/27", "Reco", True, False, False, True, 0, 512, "X", False, False, False)
    result = loop.process_date(task)
    assert result == ("/tmp/a.root", 9)
    assert called == ["readroot/read_trace.py"]


def test_process_date_with_signal_readtrace_missing_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """with_signal read stage should return 2 when executable is missing."""

    def fake_run(_cmd, check=False):  # noqa: ARG001
        raise FileNotFoundError("missing")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)

    task = loop.TaskSpec("/tmp/a.root", "2025/11/27", "Reco", True, False, False, True, 0, 512, "X", False, False, False)
    assert loop.process_date(task) == ("/tmp/a.root", 2)


def test_process_date_readheader_failure_returns_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """non-signal read stage failure should not proceed to main.py."""
    called = []

    def fake_run(cmd, check=False):  # noqa: ARG001
        called.append(cmd[1])
        if "readroot/read_header.py" in cmd:
            return SimpleNamespace(returncode=7)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(loop.subprocess, "run", fake_run)

    task = loop.TaskSpec("/tmp/a.root", "2025/11/27", "Reco", True, False, False, False, 0, 512, "X", False, False, False)
    result = loop.process_date(task)

    assert result == ("/tmp/a.root", 7)
    assert called == ["readroot/read_header.py"]


def test_process_date_readheader_missing_executable_returns_2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """non-signal read stage should return 2 when executable is missing."""

    def fake_run(_cmd, check=False):  # noqa: ARG001
        raise FileNotFoundError("missing")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)

    task = loop.TaskSpec("/tmp/a.root", "2025/11/27", "Reco", True, False, False, False, 0, 512, "X", False, False, False)
    assert loop.process_date(task) == ("/tmp/a.root", 2)


def test_process_date_success_calls_pdf_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful read+main should call PDF merge and return 0."""
    calls = {"merge": 0}

    monkeypatch.setattr(
        loop.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=0)
    )

    def fake_merge(*_args, **_kwargs):
        calls["merge"] += 1
        return True

    monkeypatch.setattr(loop, "merge_images_to_pdf", fake_merge)

    task = loop.TaskSpec("/tmp/a.root", "2025/11/27", "Reco", True, False, False, False, 0, 512, "X", False, False, False)
    result = loop.process_date(task)

    assert result == ("/tmp/a.root", 0)
    assert calls["merge"] == 1


def test_process_date_main_failure_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """main.py failure should propagate its return code."""

    def fake_run(cmd, check=False):  # noqa: ARG001
        if cmd[1] == "main.py":
            return SimpleNamespace(returncode=5)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(loop.subprocess, "run", fake_run)

    task = loop.TaskSpec("/tmp/a.root", "2025/11/27", "Reco", True, False, False, False, 0, 512, "X", False, False, False)
    assert loop.process_date(task) == ("/tmp/a.root", 5)


def test_process_date_main_missing_executable_returns_2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing main.py executable should return stable code 2."""

    def fake_run(cmd, check=False):  # noqa: ARG001
        if cmd[1] == "main.py":
            raise FileNotFoundError("missing")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(loop.subprocess, "run", fake_run)

    task = loop.TaskSpec("/tmp/a.root", "2025/11/27", "Reco", True, False, False, False, 0, 512, "X", False, False, False)
    assert loop.process_date(task) == ("/tmp/a.root", 2)


def test_process_date_pdf_merge_exception_is_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PDF merge exceptions should be logged without failing the task."""
    monkeypatch.setattr(
        loop.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(
        loop,
        "merge_images_to_pdf",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("pdf failed")),
    )

    task = loop.TaskSpec("/tmp/a.root", "2025/11/27", "Reco", True, False, False, False, 0, 512, "X", False, False, False)
    assert loop.process_date(task) == ("/tmp/a.root", 0)


def test_main_returns_1_when_base_path_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """main should return 1 when base path does not exist."""
    monkeypatch.setattr(
        loop.sys,
        "argv",
        ["loop.py", "2025-01-01T00:00:00", "2025-01-01T00:00:01", "--base-path", "/no"],
    )
    monkeypatch.setattr(loop.os.path, "exists", lambda _p: False)

    assert loop.main() == 1


def test_main_returns_2_on_any_failed_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """main should aggregate non-zero task results to exit code 2."""
    monkeypatch.setattr(
        loop.sys,
        "argv",
        [
            "loop.py",
            "2025-01-01T00:00:00",
            "2025-01-02T00:00:00",
            "--base-path",
            "/ok",
            "--limit",
            "1",
            "--run-pwm",
        ],
    )
    monkeypatch.setattr(loop.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(
        loop, "build_tasks",
        lambda *_a, **_k: [
            loop.TaskSpec("f1", "d", "o", False, False, False, False, 0, 512, "X", False, True, False),
            loop.TaskSpec("f2", "d", "o", False, False, False, False, 0, 512, "X", False, True, False),
        ],
    )
    monkeypatch.setattr(loop, "execute_tasks", lambda *_a, **_k: [("f1", 1)])

    assert loop.main() == 2


def test_main_rejects_zero_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """main should reject --jobs 0 with argparse error."""
    monkeypatch.setattr(
        loop.sys,
        "argv",
        [
            "loop.py",
            "2025-01-01T00:00:00",
            "2025-01-01T00:00:01",
            "--jobs",
            "0",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        loop.main()

    assert exc.value.code == 2


def test_main_rejects_negative_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """main should reject negative --jobs values with argparse error."""
    monkeypatch.setattr(
        loop.sys,
        "argv",
        [
            "loop.py",
            "2025-01-01T00:00:00",
            "2025-01-01T00:00:01",
            "--jobs",
            "-1",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        loop.main()

    assert exc.value.code == 2


def test_main_rejects_start_after_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """main should reject a start time that is later than end time."""
    monkeypatch.setattr(
        loop.sys,
        "argv",
        [
            "loop.py",
            "2025-01-02T00:00:00",
            "2025-01-01T00:00:00",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        loop.main()

    assert exc.value.code == 2


def test_loop_module_main_guard_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Executing loop as __main__ should raise SystemExit with main() result."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "loop.py",
            "2025-01-01T00:00:00",
            "2025-01-01T00:00:01",
            "--base-path",
            "/definitely-missing",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("loop", run_name="__main__")

    assert exc.value.code == 1