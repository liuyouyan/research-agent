"""Request-level reasoning controls must reach the final model runnable."""

from __future__ import annotations

import pytest

from research_engine import graph
from research_engine.runconfig import ROLES, RunConfig, reset_run_config, set_run_config


class _Recorder:
    def __init__(self):
        self.calls: list[dict] = []

    async def ainvoke(self, messages, **kwargs):
        self.calls.append(kwargs)
        return {"messages": messages}


@pytest.mark.asyncio
async def test_operation_override_wins_and_default_disables_forced_thinking():
    runnable = _Recorder()
    cfg = RunConfig(
        llm_mode="real",
        models={role: "custom:qwen3.8-flash" for role in ROLES},
        reasoning_effort_default="none",
        reasoning_efforts={"executor.research": "low"},
    )
    token = set_run_config(cfg)
    try:
        await graph._invoke_model(
            "executor", "executor.research", runnable, ["research"]
        )
        await graph._invoke_model(
            "executor", "executor.force_tool", runnable, ["submit"]
        )
    finally:
        reset_run_config(token)

    assert runnable.calls == [
        {"reasoning_effort": "low"},
        {"reasoning_effort": "none"},
    ]


@pytest.mark.asyncio
async def test_fake_models_never_receive_provider_specific_reasoning_kwargs():
    runnable = _Recorder()
    cfg = RunConfig(llm_mode="fake", reasoning_effort_default="low")
    token = set_run_config(cfg)
    try:
        await graph._invoke_model("executor", "executor.research", runnable, ["test"])
    finally:
        reset_run_config(token)

    assert runnable.calls == [{}]
