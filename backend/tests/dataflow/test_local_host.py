"""
The local host: SQLite cache, in-process sink, and the CLI (docs/12 M6 step 4).

This is M6's Definition of Done under test — the pipeline running with no Postgres, no
Redis, no Docker, no server and no login. The CLI tests call `main()` directly, which
calls `asyncio.run`, so they are deliberately synchronous.
"""

from __future__ import annotations

import json

import pytest

from research_engine.cli import main
from research_engine.local import InProcessEventSink, SqliteCache, run_config_from_env
from research_engine.verify_bundle import verify_file

# ── SQLite cache ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cache_roundtrips(tmp_path):
    cache = SqliteCache(tmp_path / "c.sqlite")
    assert await cache.get("missing") is None

    await cache.set("k", '["cached"]', 60)
    assert await cache.get("k") == '["cached"]'


@pytest.mark.asyncio
async def test_cache_expires_on_read(tmp_path):
    cache = SqliteCache(tmp_path / "c.sqlite")
    await cache.set("k", "v", 0)  # already expired
    assert await cache.get("k") is None
    # The expired row is cleaned up rather than left to accumulate.
    assert await cache.get("k") is None


@pytest.mark.asyncio
async def test_cache_overwrites_and_persists_across_instances(tmp_path):
    path = tmp_path / "c.sqlite"
    first = SqliteCache(path)
    await first.set("k", "one", 60)
    await first.set("k", "two", 60)

    # A separate instance on the same file — the desktop app reopening its database.
    second = SqliteCache(path)
    assert await second.get("k") == "two"


@pytest.mark.asyncio
async def test_cache_creates_parent_directory(tmp_path):
    cache = SqliteCache(tmp_path / "nested" / "deeper" / "c.sqlite")
    await cache.set("k", "v", 60)
    assert await cache.get("k") == "v"


# ── In-process sink ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sink_collects_and_forwards():
    live: list[dict] = []
    sink = InProcessEventSink(on_event=live.append)

    await sink("s", {"type": "agent_log", "agent": "planner", "message": "one"})
    await sink("s", {"type": "agent_log", "agent": "critic", "message": "two"})

    assert len(sink.events) == 2
    assert live == sink.events, "on_event fires live, in order"
    assert [e["message"] for e in sink.by_agent("planner")] == ["one"]


# ── Local config ───────────────────────────────────────────────────────────────────


def test_local_config_fake_needs_no_keys():
    config = run_config_from_env(fake=True)
    assert config.llm_mode == "fake"
    assert config.provider_keys == {}


def test_local_config_reads_env(monkeypatch):
    for env in ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-local")
    monkeypatch.setenv("MODEL_SYNTHESIZER", "anthropic:claude-sonnet-5")

    config = run_config_from_env(fake=False)

    assert config.llm_mode == "real"
    # Keys live on the config: on a laptop there is one set, so the runner's separate
    # `provider_keys` port (for a server user's BYOK override) is not needed.
    assert config.provider_keys == {"anthropic": "sk-local"}
    assert config.models["synthesizer"] == "anthropic:claude-sonnet-5"
    # Unset roles keep their defaults rather than becoming empty.
    assert config.models["planner"]


def test_local_config_reads_reasoning_and_every_role_output_limit(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("REASONING_EFFORT_DEFAULT", "none")
    monkeypatch.setenv("REASONING_EFFORT_EXECUTOR_RESEARCH", "low")
    monkeypatch.setenv("REASONING_EFFORT_SYNTHESIZER_DRAFT", "medium")

    expected_limits = {
        "planner": 2101,
        "executor": 8102,
        "critic": 3103,
        "synthesizer": 12104,
        "chat": 5105,
    }
    for role, value in expected_limits.items():
        monkeypatch.setenv(f"MAX_OUTPUT_TOKENS_{role.upper()}", str(value))

    config = run_config_from_env(fake=False)

    assert config.reasoning_for("planner.plan") == "none"
    assert config.reasoning_for("executor.research") == "low"
    assert config.reasoning_for("synthesizer.draft") == "medium"
    assert {role: config.max_output_tokens_for(role) for role in expected_limits} == (
        expected_limits
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("REASONING_EFFORT_DEFAULT", "extreme"),
        ("MAX_OUTPUT_TOKENS_EXECUTOR", "0"),
        ("MAX_OUTPUT_TOKENS_CRITIC", "not-a-number"),
    ],
)
def test_local_config_rejects_invalid_generation_controls(monkeypatch, name, value):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        run_config_from_env(fake=False)


def test_local_config_without_any_key_says_what_to_do(monkeypatch):
    for env in (
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "CUSTOM_API_KEY",
        "CUSTOM_BASE_URL",
        "MODEL_PLANNER",
        "MODEL_EXECUTOR",
        "MODEL_CRITIC",
        "MODEL_SYNTHESIZER",
        "MODEL_CHAT",
    ):
        monkeypatch.delenv(env, raising=False)

    with pytest.raises(SystemExit) as excinfo:
        run_config_from_env(fake=False)

    message = str(excinfo.value)
    assert "GOOGLE_API_KEY" in message
    assert "--fake" in message, "the error should point at the keyless path"


# ── The CLI: M6's Definition of Done ───────────────────────────────────────────────


def test_cli_runs_to_the_gate_then_approves_from_the_checkpoint(tmp_path, capsys):
    """The whole promise: a cited report and a working review gate, on a bare machine."""
    code = main(
        [
            "Why do LLMs hallucinate?",
            "--fake",
            "--quiet",
            "--session-id",
            "t1",
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    gate = capsys.readouterr().out
    assert "awaiting_approval" in gate
    assert "[1]" in gate, "the draft carries inline citations"
    assert "--approve t1" in gate, "the session id must be recoverable from the output"

    # A second `main()` is a second `asyncio.run` and a second saver — the state comes
    # back from the SQLite file, not from memory.
    code = main(["--approve", "t1", "--fake", "--quiet", "--data-dir", str(tmp_path)])
    assert code == 0
    assert "completed" in capsys.readouterr().out

    assert (tmp_path / "checkpoints.sqlite").exists()
    assert (tmp_path / "cache.sqlite").exists()


def test_cli_rework_loop(tmp_path, capsys):
    main(["q", "--fake", "--quiet", "--session-id", "t2", "--data-dir", str(tmp_path)])
    capsys.readouterr()

    code = main(
        [
            "--reject",
            "t2",
            "-f",
            "Add limitations.",
            "--fake",
            "--quiet",
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert "awaiting_approval" in capsys.readouterr().out, "rework returns to the gate"

    assert (
        main(["--approve", "t2", "--fake", "--quiet", "--data-dir", str(tmp_path)]) == 0
    )
    assert "completed" in capsys.readouterr().out


def test_cli_json_output_is_machine_readable(tmp_path, capsys):
    code = main(
        [
            "q",
            "--fake",
            "--quiet",
            "--yes",
            "--json",
            "--session-id",
            "t3",
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["report"]
    assert len(payload["sources"]) >= 1
    assert payload["cost_usd"] > 0
    assert payload["elapsed_seconds"] is not None
    manifest = json.loads(
        (tmp_path / "sessions" / "t3.json").read_text(encoding="utf-8")
    )
    assert manifest["approval_chain"][-1]["action"] == "approved"


def test_cli_exports_and_verifies_a_formal_bundle_without_mutating_session(
    tmp_path, capsys
):
    """Approval is durable; export reads session state and writes only its target artifact."""
    assert (
        main(
            [
                "q",
                "--fake",
                "--quiet",
                "--session-id",
                "bundle-1",
                "--data-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            ["--approve", "bundle-1", "--fake", "--quiet", "--data-dir", str(tmp_path)]
        )
        == 0
    )
    capsys.readouterr()

    manifest_path = tmp_path / "sessions" / "bundle-1.json"
    manifest_before = manifest_path.read_text(encoding="utf-8")
    output_path = tmp_path / "research.bundle.json"
    assert (
        main(
            [
                "--export-bundle",
                "bundle-1",
                "--bundle-output",
                str(output_path),
                "--data-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert output_path.exists()
    assert manifest_path.read_text(encoding="utf-8") == manifest_before
    capsys.readouterr()

    result = verify_file(output_path)
    assert result.passed, [check for check in result.checks if not check.passed]
    assert main(["--verify-bundle", str(output_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_cli_streams_progress_unless_quiet(tmp_path, capsys):
    main(["q", "--fake", "--yes", "--session-id", "t4", "--data-dir", str(tmp_path)])
    captured = capsys.readouterr()

    # Progress goes to stderr so stdout stays a clean, pipeable report.
    assert "planner" in captured.err
    assert "synthesizer" in captured.err
    assert "planner" not in captured.out


def test_cli_requires_a_query_or_a_session(capsys):
    with pytest.raises(SystemExit):
        main(["--fake"])
    assert "query" in capsys.readouterr().err


def test_cli_reject_requires_feedback(capsys):
    with pytest.raises(SystemExit):
        main(["--reject", "whatever", "--fake"])
    assert "feedback" in capsys.readouterr().err
