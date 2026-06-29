"""Config loading, dotted/file attribute import, judge resolution, and suite building."""

import pytest

from agent_eval.config.loader import build_suite, import_attr, load_config, resolve_judge
from agent_eval.config.schema import ConfigModel
from agent_eval.core.errors import ConfigError
from agent_eval.core.registry import default_registry

_FACTORY_FILE = """
from agent_eval.judges.backend import FunctionJudge
single = FunctionJudge(lambda req: 0.42)
panel = [FunctionJudge(lambda req: 0.4), FunctionJudge(lambda req: 0.6)]
"""


def _write_config(tmp_path, body: str):
    p = tmp_path / "c.yaml"
    p.write_text(body)
    return p


def test_load_config_expands_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DSPATH", "/data/x.jsonl")
    body = "agent_type: rag\ndatasets:\n  d:\n    adapter: jsonl\n    path: ${DSPATH}\n"
    cfg = load_config(_write_config(tmp_path, body))
    assert cfg.datasets["d"].path == "/data/x.jsonl"


def test_import_attr_module_form():
    assert import_attr("math:sqrt")(4) == 2.0


def test_import_attr_file_form(tmp_path):
    f = tmp_path / "fac.py"
    f.write_text(_FACTORY_FILE)
    judge = import_attr(f"{f}:single")
    assert judge.evaluate.__self__ is judge  # it's a FunctionJudge instance


def test_import_attr_bad_spec():
    with pytest.raises(ConfigError):
        import_attr("no-colon-no-dot")


def test_resolve_judge_none_when_unset():
    assert resolve_judge(ConfigModel(agent_type="plain")) == (None, ())


def test_resolve_judge_single(tmp_path):
    f = tmp_path / "fac.py"
    f.write_text(_FACTORY_FILE)
    cfg = ConfigModel(agent_type="rag", judge={"factory": f"{f}:single"})
    backend, panel = resolve_judge(cfg)
    assert backend is not None and panel == ()


def test_resolve_judge_panel(tmp_path):
    f = tmp_path / "fac.py"
    f.write_text(_FACTORY_FILE)
    cfg = ConfigModel(agent_type="rag", judge={"factory": f"{f}:panel"})
    backend, panel = resolve_judge(cfg)
    assert backend is not None and len(panel) == 1  # first is primary, rest is panel


def test_build_suite_injects_judge_and_defaults(tmp_path):
    f = tmp_path / "fac.py"
    f.write_text(_FACTORY_FILE)
    cfg = ConfigModel(
        agent_type="rag",
        judge={"factory": f"{f}:single"},
        defaults={"k": 3},
        suites={"s": {"metrics": ["recall_at_k", "faithfulness"],
                      "gate": {"thresholds": {"recall_at_k": 0.5}}}},
    )
    suite = build_suite(cfg, "s", default_registry())
    names = {m.name for m in suite.metrics}
    assert names == {"recall_at_k", "faithfulness"}
    recall = next(m for m in suite.metrics if m.name == "recall_at_k")
    assert recall.k == 3  # default injected
