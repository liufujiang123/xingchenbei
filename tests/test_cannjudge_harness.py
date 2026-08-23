import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_platform_pass_is_terminal_and_successful():
    evaluator = load_module("cannjudge_eval_test", "tools/cannjudge_eval.py")
    harness = load_module("agent_loop_test", "tools/agent_loop.py")

    assert "Pass" in evaluator.TERMINAL_STATUSES
    assert "Pass" in harness.PLATFORM_PASS_STATUSES
    assert "Accepted" in evaluator.TERMINAL_STATUSES
    assert "Accepted" in harness.PLATFORM_PASS_STATUSES
