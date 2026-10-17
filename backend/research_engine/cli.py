"""
`research-engine` — a reference local host (docs/12 M6 step 4, its Definition of Done).

Runs the full pipeline on a bare machine: no Docker, no Postgres, no Redis, no login, no
server. State lives in one SQLite file, so the review gate works the way it should —
the run pauses, the process exits, and a later `--approve` resumes *from the checkpoint*
rather than re-running research you already paid for.

    research-engine "why do LLMs hallucinate?" --fake
    research-engine --approve <session-id>            # finalize the draft
    research-engine --reject <session-id> -f "cite more primary sources"
    research-engine --export-draft <session-id> --output draft.md
    research-engine --export-bundle <session-id> --bundle-output report.bundle.json
    research-engine --verify-bundle report.bundle.json

The extended host can also review the Planner interrupt, manage a local corpus, inspect
checkpoint evidence, persist sanitized event details, select Web or corpus-only evidence,
and atomically export a paused draft or completed report. These are host concerns wired
through existing runner ports and RunConfig fields; the graph is unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import uuid
from collections.abc import Mapping
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_engine import bundle, verify_bundle
from research_engine.checkpoint_read import CheckpointOutcome, read_checkpoint
from research_engine.corpus import CorpusStore
from research_engine.embeddings import FakeEmbeddings, LocalEmbeddings
from research_engine.local import (
    InProcessEventSink,
    SqliteCache,
    default_data_dir,
    load_env_file,
    run_config_from_env,
)
from research_engine.runner import RunOutcome, resume, run
from research_engine.schemas import OutlineSection, ResearchTask

_MANIFEST_VERSION = 3
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class _RuntimeOptions:
    fake: bool
    source: str
    corpus_db: Path
    plan_gate: bool
    embedding_model: str
    embedding_base_url: str
    embedding_dimensions: int
    output_path: Path | None
    event_log_path: Path | None


@dataclass
class _CliRun:
    outcome: RunOutcome
    session_id: str
    data_dir: Path
    source: str
    corpus_db: Path | None = None
    plan_file: Path | None = None
    output_path: Path | None = None
    event_log_path: Path | None = None
    output_written: bool = False
    output_error: str | None = None


# ── Output ─────────────────────────────────────────────────────────────────────────

_AGENT_MARK = {
    "planner": "◆",
    "executor": "▸",
    "critic": "✓",
    "synthesizer": "■",
}

_SENSITIVE_FIELDS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "refresh_token",
    "secret",
    "token",
}


def _is_sensitive_field(name: object) -> bool:
    normalized = str(name).casefold().replace("-", "_")
    return normalized in _SENSITIVE_FIELDS or normalized.endswith(
        ("_api_key", "_password", "_secret", "_token")
    )


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _is_sensitive_field(key) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def _print_event(event: Mapping[str, Any], *, verbose: bool = False) -> None:
    agent = event.get("agent") or "-"
    message = event.get("message") or ""
    print(f"  {_AGENT_MARK.get(agent, '·')} {agent:<12} {message}", file=sys.stderr)
    detail = event.get("detail")
    if verbose and detail not in (None, {}, []):
        rendered = json.dumps(detail, ensure_ascii=False, indent=2, default=str)
        print("\n".join(f"      {line}" for line in rendered.splitlines()), file=sys.stderr)


class _EventReporter:
    """Fan one sanitized event stream out to the terminal and an optional JSONL file."""

    def __init__(
            self,
            *,
            session_id: str,
            quiet: bool,
            verbose: bool,
            path: Path | None,
            new_session: bool,
    ) -> None:
        self.session_id = session_id
        self.quiet = quiet
        self.verbose = verbose
        self.path = path
        self.new_session = new_session
        self._handle = None

    def __enter__(self) -> _EventReporter:
        if self.path is None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_dir():
            raise ValueError(f"Event log path is a directory: {self.path}")
        mode = "x" if self.new_session else "a"
        try:
            self._handle = self.path.open(mode, encoding="utf-8", newline="\n")
        except FileExistsError as exc:
            raise FileExistsError(
                f"Event log already exists: {self.path}. Choose a new path for a new session."
            ) from exc
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc is not None and self._handle is not None:
            self.lifecycle("Command failed", status="error", error=str(exc))
        if self._handle is not None:
            self._handle.close()

    def __call__(self, event: dict) -> None:
        sanitized = _redact(event)
        if not self.quiet:
            _print_event(sanitized, verbose=self.verbose)
        if self._handle is None:
            return
        record = {"session_id": self.session_id, **sanitized}
        self._handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def lifecycle(self, message: str, **detail: Any) -> None:
        self(
            {
                "type": "cli_lifecycle",
                "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "agent": "cli",
                "message": message,
                "detail": detail or None,
                "data": None,
            }
        )


_SOURCES_HEADING_RE = re.compile(
    r"^#{1,6}\s*(sources|references|citations|bibliography)\b", re.I | re.M
)


def _report_with_sources(report: str | None, sources: list[dict]) -> str | None:
    """Append the outcome source list unless the report already has one."""
    if not report or not sources or _SOURCES_HEADING_RE.search(report):
        return report

    lines = ["## Sources"]
    for source in sources:
        lines.append(f"[{source.get('index')}] {source.get('title') or '(untitled)'}")
        lines.append(f"    {source.get('url') or ''}")

    return f"{report.rstrip()}\n\n" + "\n".join(lines) + "\n"


def _command_data_dir(run_result: _CliRun) -> str:
    default = default_data_dir().expanduser().resolve()
    return "" if run_result.data_dir == default else f' --data-dir "{run_result.data_dir}"'


def _print_outcome(run_result: _CliRun) -> None:
    outcome = run_result.outcome
    # report = outcome.report
    report = _report_with_sources(outcome.report, outcome.sources)
    if report:
        print()
        print(report)

    # if report and outcome.sources and not _SOURCES_HEADING_RE.search(report):
    #     print()
    #     print("## Sources")
    #     for source in outcome.sources:
    #         print(f"[{source.get('index')}] {source.get('title') or '(untitled)'}")
    #         print(f"    {source.get('url')}")

    print()
    print(
        f"— {outcome.status} · {len(outcome.sources)} sources · "
        f"${outcome.cost_usd:.4f} · {outcome.tokens_input + outcome.tokens_output} tokens"
        + (f" · {outcome.elapsed_seconds}s" if outcome.elapsed_seconds is not None else "")
    )

    data_arg = _command_data_dir(run_result)
    if outcome.status == "awaiting_approval":
        print()
        print("Paused at the report review gate. Nothing is final until you approve it:")
        print(f"  research-engine --approve {run_result.session_id}{data_arg}")
        print(f'  research-engine --reject {run_result.session_id} -f "what to fix"{data_arg}')
    elif outcome.status == "awaiting_plan":
        print()
        print("Paused at the research design gate before retrieval begins.")
        if run_result.plan_file is not None:
            print(f"Editable plan: {run_result.plan_file}")
            print(
                f"  research-engine --resume-plan {run_result.session_id} "
                f'--plan-file "{run_result.plan_file}"{data_arg}'
            )
        print(f"  research-engine --resume-plan {run_result.session_id}{data_arg}")
    elif outcome.status == "failed":
        print(f"\nFailed: {outcome.error}", file=sys.stderr)

    if run_result.output_path is not None:
        if run_result.output_written:
            print(f"\nReport written to: {run_result.output_path}")
        elif run_result.output_error:
            prefix = (
                "Research completed, but the report could not be written"
                if outcome.status == "completed"
                else "Report could not be written"
            )
            print(f"\n{prefix}: {run_result.output_error}", file=sys.stderr)
        elif outcome.status != "completed":
            print(f"\nReport output pending: {run_result.output_path}")
    if run_result.event_log_path is not None:
        print(f"\nEvent log: {run_result.event_log_path}")


def _run_payload(run_result: _CliRun) -> dict[str, Any]:
    outcome = run_result.outcome
    return {
        "session_id": run_result.session_id,
        "status": outcome.status,
        "source": run_result.source,
        "corpus_db": str(run_result.corpus_db) if run_result.corpus_db else None,
        "report": outcome.report,
        "sources": outcome.sources,
        "plan_tasks": outcome.plan_tasks,
        "plan_outline": outcome.plan_outline,
        "plan_file": str(run_result.plan_file) if run_result.plan_file else None,
        "output_path": str(run_result.output_path) if run_result.output_path else None,
        "event_log_path": (str(run_result.event_log_path) if run_result.event_log_path else None),
        "output_written": run_result.output_written,
        "output_error": run_result.output_error,
        "cost_usd": outcome.cost_usd,
        "tokens_input": outcome.tokens_input,
        "tokens_output": outcome.tokens_output,
        "rework_count": outcome.rework_count,
        "elapsed_seconds": outcome.elapsed_seconds,
        "error": outcome.error,
    }


# ── Durable local metadata ─────────────────────────────────────────────────────────


def _data_dir(args: argparse.Namespace) -> Path:
    path = Path(args.data_dir).expanduser() if args.data_dir else default_data_dir()
    return path.resolve()


def _absolute_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _validate_session_id(session_id: str) -> str:
    if not _SESSION_ID_RE.fullmatch(session_id):
        raise ValueError(
            "Session ids may contain only letters, digits, '.', '_' and '-', must start "
            "with a letter or digit, and may be at most 128 characters."
        )
    return session_id


def _atomic_write_text(path: Path, text: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_dir():
            raise ValueError(f"Output path is a directory: {path}")
        if not overwrite:
            raise FileExistsError(
                f"Output file already exists: {path}. Use --overwrite-output to replace it."
            )

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        overwrite=True,
    )


def _manifest_path(data_dir: Path, session_id: str) -> Path:
    return data_dir / "sessions" / f"{_validate_session_id(session_id)}.json"


def _load_manifest(data_dir: Path, session_id: str) -> dict[str, Any] | None:
    path = _manifest_path(data_dir, session_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read session manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Session manifest is not a JSON object: {path}")
    if payload.get("session_id") not in (None, session_id):
        raise ValueError(f"Session manifest {path} belongs to another session.")
    return payload


# 获取一个审批链
def _approval_chain_from_manifest(manifest: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Return validated local approval records, preserving their original order.

    The manifest is the local host's durable audit store. Export must not invent an
    approval after the fact, so malformed historical data is an error rather than a
    value that can silently turn into a plausible-looking bundle.
    """
    raw_chain = (manifest or {}).get("approval_chain", [])
    if raw_chain is None:
        return []
    if not isinstance(raw_chain, list):
        raise TypeError("Session manifest approval_chain must be a JSON array.")

    chain: list[dict[str, Any]] = []
    for position, raw_record in enumerate(raw_chain, start=1):
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"Approval record {position} must be a JSON object.")
        action = raw_record.get("action")
        draft_hash = raw_record.get("draft_hash")
        timestamp = raw_record.get("timestamp")
        feedback = raw_record.get("feedback")
        if not all(isinstance(value, str) and value for value in (action, draft_hash, timestamp)):
            raise ValueError(
                f"Approval record {position} needs non-empty action, draft_hash, and timestamp."
            )
        if feedback is not None and not isinstance(feedback, str):
            raise ValueError(f"Approval record {position} feedback must be a string or null.")
        record = {"action": action, "draft_hash": draft_hash, "timestamp": timestamp}
        if feedback is not None:
            record["feedback"] = feedback
        chain.append(record)
    return chain


# 创建一个审批记录
def _approval_record(
        *, action: str, draft_hash: str, feedback: str | None = None
) -> dict[str, Any]:
    """Create a durable local report-review record."""
    record: dict[str, Any] = {
        "action": action,
        "draft_hash": draft_hash,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if feedback is not None:
        record["feedback"] = feedback
    return record


def _write_manifest(
        data_dir: Path,
        session_id: str,
        options: _RuntimeOptions,
        models: Mapping[str, str],
        approval_chain: list[dict[str, Any]],
) -> None:
    _write_json(
        _manifest_path(data_dir, session_id),
        {
            "version": _MANIFEST_VERSION,
            "session_id": session_id,
            "fake": options.fake,
            "source": options.source,
            "corpus_db": str(options.corpus_db),
            "plan_gate": options.plan_gate,
            "embedding": {
                "model": options.embedding_model,
                "base_url": options.embedding_base_url,
                "dimensions": options.embedding_dimensions,
            },
            "models": dict(models),
            "output_path": str(options.output_path) if options.output_path else None,
            "event_log_path": (str(options.event_log_path) if options.event_log_path else None),
            "approval_chain": approval_chain,
        },
    )

# 在“用户这次显式传入的配置”和“之前 session 保存下来的配置”之间做合并，而且如果两者冲突，就拒绝继续恢复该 session。
def _saved_or_explicit(
        name: str,
        explicit: Any,
        saved: Any,
        default: Any,
        *,
        compare=lambda value: value,
) -> Any:
    if explicit is not None and saved is not None and compare(explicit) != compare(saved):
        raise ValueError(
            f"{name} conflicts with the value recorded for this session "
            f"({saved!r}); resume with the original configuration."
        )
    if explicit is not None:
        return explicit
    return saved if saved is not None else default


def _runtime_options(
        args: argparse.Namespace,
        data_dir: Path,
        manifest: Mapping[str, Any] | None = None,
) -> _RuntimeOptions:
    saved = manifest or {}
    saved_embedding = saved.get("embedding")
    if not isinstance(saved_embedding, Mapping):
        saved_embedding = {}

    fake = bool(_saved_or_explicit("--fake", args.fake, saved.get("fake"), False))
    source = _saved_or_explicit("--source", args.source, saved.get("source"), "web")

    explicit_corpus = _absolute_path(args.corpus_db) if args.corpus_db else None
    saved_corpus = _absolute_path(saved["corpus_db"]) if saved.get("corpus_db") else None
    corpus_db = _saved_or_explicit(
        "--corpus-db",
        explicit_corpus,
        saved_corpus,
        (data_dir / "corpus.sqlite").resolve(),
        compare=str,
    )

    plan_default = bool(args.resume_plan)
    plan_gate = bool(saved.get("plan_gate", args.plan_gate or plan_default))

    if fake:
        if any(
                value is not None
                for value in (
                        args.embedding_model,
                        args.embedding_base_url,
                        args.embedding_dimensions,
                )
        ):
            raise ValueError(
                "--fake uses deterministic in-process embeddings; do not combine it "
                "with embedding endpoint options."
            )
        embedding_model = "local:fake-deterministic"
        embedding_base_url = ""
        embedding_dimensions = 768
    else:
        embedding_model = _saved_or_explicit(
            "--embedding-model",
            args.embedding_model,
            saved_embedding.get("model"),
            os.environ.get("CORPUS_EMBEDDINGS_MODEL", "nomic-embed-text"),
        )
        embedding_base_url = _saved_or_explicit(
            "--embedding-base-url",
            args.embedding_base_url,
            saved_embedding.get("base_url"),
            os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )
        embedding_dimensions = int(
            _saved_or_explicit(
                "--embedding-dimensions",
                args.embedding_dimensions,
                saved_embedding.get("dimensions"),
                768,
                compare=int,
            )
        )
        if embedding_dimensions <= 0:
            raise ValueError("--embedding-dimensions must be greater than zero")

    # Output is an artifact destination rather than research configuration. Supplying a
    # new path while resuming is therefore an intentional "save as", not a conflict.
    output_path = (
        _absolute_path(args.output)
        if args.output
        else _absolute_path(saved["output_path"])
        if saved.get("output_path")
        else None
    )
    if output_path is not None and output_path.suffix.lower() != ".md":
        raise ValueError("--output currently supports only Markdown (.md) files")

    explicit_event_log = _absolute_path(args.event_log) if args.event_log else None
    saved_event_log = (
        _absolute_path(saved["event_log_path"]) if saved.get("event_log_path") else None
    )
    event_log_path = _saved_or_explicit(
        "--event-log",
        explicit_event_log,
        saved_event_log,
        None,
        compare=lambda value: os.path.normcase(str(value)),
    )
    if event_log_path is not None and event_log_path.suffix.lower() != ".jsonl":
        raise ValueError("--event-log must use a .jsonl file")

    return _RuntimeOptions(
        fake=fake,
        source=source,
        corpus_db=corpus_db,
        plan_gate=plan_gate,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
        embedding_dimensions=embedding_dimensions,
        output_path=output_path,
        event_log_path=event_log_path,
    )


# ── Corpus management ──────────────────────────────────────────────────────────────


def _embedder(options: _RuntimeOptions):
    if options.fake:
        return FakeEmbeddings()
    return LocalEmbeddings(
        model=options.embedding_model,
        base_url=options.embedding_base_url,
        dimensions=options.embedding_dimensions,
    )


def _corpus_store(options: _RuntimeOptions) -> CorpusStore:
    return CorpusStore(options.corpus_db, _embedder(options))


async def _require_searchable_corpus(store: CorpusStore) -> None:
    documents = await store.documents()
    searchable = [
        row for row in documents if row.get("origin") != "generated" and row.get("is_current", True)
    ]
    if not searchable:
        raise ValueError(
            "The corpus has no current uploaded documents. Use --ingest before "
            "starting corpus research."
        )

    status = await store.status()
    current_model = status["current_model"]
    chunks_by_model = status["chunks_by_model"]
    if not chunks_by_model.get(current_model):
        indexed = ", ".join(sorted(chunks_by_model)) or "none"
        raise ValueError(
            f"The corpus has no chunks for embedding model '{current_model}'. "
            f"Indexed models: {indexed}. Re-ingest with the selected model."
        )


async def _manage_corpus(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = _data_dir(args)
    data_dir.mkdir(parents=True, exist_ok=True)
    load_env_file()
    options = _runtime_options(args, data_dir)
    store = _corpus_store(options)

    if args.ingest:
        ingested: list[dict[str, Any]] = []
        for value in args.ingest:
            path = _absolute_path(value)
            if not path.is_file():
                raise ValueError(f"Document does not exist or is not a file: {path}")
            # CorpusStore logs successful ingestion to stdout under the standalone
            # logger. JSON mode reserves stdout for one parseable document.
            output_guard = redirect_stdout(sys.stderr) if args.json else nullcontext()
            with output_guard:
                result = await store.ingest(path.name, await asyncio.to_thread(path.read_bytes))
            ingested.append(
                {
                    "path": str(path),
                    "doc_id": result.doc_id,
                    "doc_key": result.doc_key,
                    "version": result.version,
                    "filename": result.filename,
                    "chunks_written": result.chunks_written,
                    "skipped": result.skipped,
                    "reason": result.reason,
                }
            )
        return {
            "action": "ingest",
            "corpus_db": str(options.corpus_db),
            "documents": ingested,
        }

    if args.list_corpus:
        return {
            "action": "list",
            "corpus_db": str(options.corpus_db),
            "documents": await store.documents(),
        }

    if args.corpus_status:
        return {
            "action": "status",
            "corpus_db": str(options.corpus_db),
            "status": await store.status(),
        }

    deleted = await store.delete(args.delete_document)
    if not deleted:
        raise ValueError(f"Document not found in corpus: {args.delete_document}")
    return {
        "action": "delete",
        "corpus_db": str(options.corpus_db),
        "document_id": args.delete_document,
        "deleted": True,
    }


def _print_corpus_result(result: Mapping[str, Any]) -> None:
    action = result["action"]
    if action == "ingest":
        for document in result["documents"]:
            if document["skipped"]:
                print(f"Skipped {document['path']}: {document['reason'] or 'already indexed'}")
            else:
                print(
                    f"Ingested {document['filename']} · {document['chunks_written']} chunks "
                    f"· id {document['doc_id']}"
                )
    elif action == "list":
        documents = result["documents"]
        if not documents:
            print("The corpus is empty.")
        for document in documents:
            current = "current" if document.get("is_current", True) else "superseded"
            print(
                f"{document['id']} · {document['filename']} · "
                f"{document['chunk_count']} chunks · {current}"
            )
    elif action == "status":
        print(json.dumps(result["status"], ensure_ascii=False, indent=2))
    else:
        print(f"Deleted document {result['document_id']}.")
    print(f"Corpus: {result['corpus_db']}")


# ── Plan review, checkpoint inspection, and export ────────────────────────────────


def _plan_path(data_dir: Path, session_id: str) -> Path:
    return data_dir / "plans" / f"{_validate_session_id(session_id)}.json"


def _write_plan(data_dir: Path, session_id: str, outcome: RunOutcome) -> Path:
    path = _plan_path(data_dir, session_id)
    _write_json(
        path,
        {
            "session_id": session_id,
            "tasks": outcome.plan_tasks,
            "outline": outcome.plan_outline,
        },
    )
    return path


def _load_plan(path_value: str, session_id: str) -> dict[str, list[dict]]:
    path = _absolute_path(path_value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read plan file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("The plan file must contain a JSON object.")
    if payload.get("session_id") not in (None, session_id):
        raise ValueError(f"Plan file {path} belongs to another session.")

    decision: dict[str, list[dict]] = {}
    if "tasks" in payload:
        if not isinstance(payload["tasks"], list):
            raise ValueError("Plan 'tasks' must be a JSON array.")
        try:
            tasks = [
                ResearchTask.model_validate(row).model_dump(mode="json") for row in payload["tasks"]
            ]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Plan contains an invalid research task: {exc}") from exc
        if not any(task["include"] for task in tasks):
            raise ValueError("The plan must include at least one research task.")
        decision["tasks"] = tasks

    if "outline" in payload:
        if not isinstance(payload["outline"], list):
            raise ValueError("Plan 'outline' must be a JSON array.")
        try:
            decision["outline"] = [
                OutlineSection.model_validate(row).model_dump(mode="json")
                for row in payload["outline"]
            ]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Plan contains an invalid outline section: {exc}") from exc
    return decision


def _checkpoint_phase(values: Mapping[str, Any]) -> str:
    if values.get("final_report"):
        return "completed"
    if values.get("error"):
        return "failed"
    if values.get("draft_report"):
        return "awaiting_approval"
    if values.get("tasks") and not values.get("plan_approved"):
        return "awaiting_plan"
    return "unknown"


async def _checked_checkpoint(saver, session_id: str):
    snapshot = await read_checkpoint(saver, session_id)
    if snapshot.outcome == CheckpointOutcome.MISSING:
        raise ValueError(f"Session '{session_id}' was not found in the checkpoint database.")
    if snapshot.outcome == CheckpointOutcome.UNREADABLE:
        raise ValueError(f"Session '{session_id}' has an unreadable checkpoint: {snapshot.error}")
    return snapshot


async def _inspect_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    data_dir = _data_dir(args)
    checkpoint_db = data_dir / "checkpoints.sqlite"
    if not checkpoint_db.is_file():
        raise ValueError(f"Checkpoint database was not found: {checkpoint_db}")

    session_id = _validate_session_id(args.inspect)
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_db)) as saver:
        snapshot = await _checked_checkpoint(saver, session_id)

    values = snapshot.values
    selected = (
        ("evidence", "sources", "contradictions") if (args.show or "all") == "all" else (args.show,)
    )
    payload: dict[str, Any] = {
        "session_id": session_id,
        "status": _checkpoint_phase(values),
        "counts": {
            "evidence": len(values.get("evidence") or []),
            "sources": len(values.get("sources") or []),
            "contradictions": len(values.get("contradictions") or []),
        },
    }
    for field in selected:
        payload[field] = values.get(field) or []
    return payload


def _print_inspection(payload: Mapping[str, Any]) -> None:
    print(f"Session: {payload['session_id']}")
    print(f"Status: {payload['status']}")
    counts = payload["counts"]
    print(
        f"Evidence: {counts['evidence']} · Sources: {counts['sources']} · "
        f"Contradictions: {counts['contradictions']}"
    )
    for field in ("evidence", "sources", "contradictions"):
        if field not in payload:
            continue
        print(f"\n## {field.title()}")
        print(json.dumps(payload[field], ensure_ascii=False, indent=2, default=str))


def _checkpoint_records(values: Mapping[str, Any], name: str) -> list[dict]:
    """Read a list of object records from a serialized graph checkpoint."""
    raw_records = values.get(name) or []
    if not isinstance(raw_records, list) or not all(isinstance(row, Mapping) for row in raw_records):
        raise ValueError(f"Checkpoint field '{name}' must be a list of objects.")
    return [dict(row) for row in raw_records]


def _read_event_trace(manifest: Mapping[str, Any], session_id: str) -> tuple[list[dict], bool]:
    """Read the optional, sanitized JSONL event log without changing it.

    A session without ``--event-log`` truthfully exports ``trace_available=False``.
    A configured but unreadable log is an export error: calling it unavailable would
    conceal an artifact that was supposed to have been retained.
    """
    path_value = manifest.get("event_log_path")
    if path_value in (None, ""):
        return [], False
    if not isinstance(path_value, str):
        raise TypeError("Session manifest event_log_path must be a string or null.")

    path = _absolute_path(path_value)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read event log {path}: {exc}") from exc

    trace: list[dict] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in event log {path} at line {line_number}: {exc}") from exc
        if not isinstance(record, Mapping):
            raise TypeError(f"Event log {path} line {line_number} is not a JSON object.")
        if record.get("session_id") == session_id:
            trace.append(dict(record))
    return trace, True


async def _export_bundle(args: argparse.Namespace) -> str:
    """Assemble a completed local session into a verifier-approved bundle.

    This deliberately follows ``_inspect_checkpoint`` rather than ``_drive``: it never
    creates directories, calls ``saver.setup()``, resumes the graph, or writes the session
    manifest. The only optional write happens later when ``--bundle-output`` is requested.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    data_dir = _data_dir(args)
    checkpoint_db = data_dir / "checkpoints.sqlite"
    if not checkpoint_db.is_file():
        raise ValueError(f"Checkpoint database was not found: {checkpoint_db}")

    session_id = _validate_session_id(args.export_bundle)
    manifest = _load_manifest(data_dir, session_id)
    if manifest is None:
        raise ValueError(
            f"Session '{session_id}' predates CLI manifests and has no durable approval chain. "
            "It cannot be exported as a formal bundle."
        )

    approval_chain = _approval_chain_from_manifest(manifest)
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_db)) as saver:
        snapshot = await _checked_checkpoint(saver, session_id)

    values = snapshot.values
    phase = _checkpoint_phase(values)
    if phase != "completed":
        raise ValueError(
            f"The session is {phase}; --export-bundle requires a completed, approved session."
        )
    report = values.get("final_report")
    query = values.get("original_query")
    if not isinstance(report, str) or not report:
        raise ValueError("The completed session has no final report to bundle.")
    if not isinstance(query, str) or not query:
        raise ValueError("The completed session has no original query to bundle.")

    trace, trace_available = _read_event_trace(manifest, session_id)
    research_depth = values.get("research_depth")
    model_routes = manifest.get("models") or {}
    if not isinstance(model_routes, Mapping):
        raise TypeError("Session manifest models must be a JSON object.")
    manifest_bundle = bundle.assemble(
        session_id=session_id,
        query=query,
        report=report,
        evidence=_checkpoint_records(values, "evidence"),
        sources=_checkpoint_records(values, "sources"),
        contradictions=_checkpoint_records(values, "contradictions"),
        models={str(role): str(route) for role, route in model_routes.items()},
        cost_usd=float(values.get("cost_usd") or 0.0),
        tokens_input=int(values.get("tokens_input") or 0),
        tokens_output=int(values.get("tokens_output") or 0),
        # Completion duration is not persisted in the local checkpoint. ``null`` is
        # intentional: measuring from export time would falsely imply research took longer.
        elapsed_seconds=None,
        research_depth=research_depth if isinstance(research_depth, str) else "balanced",
        approval_chain=approval_chain,
        trace=trace,
        trace_available=trace_available,
        demo=bool(manifest.get("fake", False)),
    )
    verification = verify_bundle.verify(manifest_bundle)
    if not verification.passed:
        failed = ", ".join(check.name for check in verification.checks if not check.passed)
        raise ValueError(
            f"Session '{session_id}' cannot be exported as a formal bundle; verification failed: "
            f"{failed}."
        )
    return bundle.serialize(manifest_bundle)


def _outcome_from_checkpoint(values: Mapping[str, Any]) -> RunOutcome:
    report = values.get("final_report")
    if not isinstance(report, str) or not report:
        raise ValueError("The session is not completed and has no final report to export.")
    return RunOutcome(
        status="completed",
        draft_report=values.get("draft_report"),
        final_report=report,
        sources=list(values.get("sources") or []),
        cost_usd=round(float(values.get("cost_usd") or 0.0), 6),
        tokens_input=int(values.get("tokens_input") or 0),
        tokens_output=int(values.get("tokens_output") or 0),
        rework_count=int(values.get("rework_count") or 0),
    )


def _draft_outcome_from_checkpoint(values: Mapping[str, Any]) -> RunOutcome:
    phase = _checkpoint_phase(values)
    if phase != "awaiting_approval":
        raise ValueError(
            f"The session is {phase}; --export-draft requires a session awaiting approval."
        )
    report = values.get("draft_report")
    if not isinstance(report, str) or not report:
        raise ValueError("The session has no paused draft to export.")
    return RunOutcome(
        status="awaiting_approval",
        draft_report=report,
        sources=list(values.get("sources") or []),
        cost_usd=round(float(values.get("cost_usd") or 0.0), 6),
        tokens_input=int(values.get("tokens_input") or 0),
        tokens_output=int(values.get("tokens_output") or 0),
        rework_count=int(values.get("rework_count") or 0),
    )


def _write_completed_output(run_result: _CliRun, *, overwrite: bool) -> None:
    if run_result.outcome.status != "completed" or run_result.output_path is None:
        return
    # report = run_result.outcome.final_report
    report = _report_with_sources(
        run_result.outcome.final_report,
        run_result.outcome.sources,
    )
    if not report:
        run_result.output_error = "the completed checkpoint contains no final report"
        return
    try:
        _atomic_write_text(run_result.output_path, report, overwrite=overwrite)
    except (OSError, ValueError) as exc:
        run_result.output_error = str(exc)
    else:
        run_result.output_written = True


def _write_draft_output(run_result: _CliRun, *, overwrite: bool) -> None:
    if run_result.outcome.status != "awaiting_approval" or run_result.output_path is None:
        return
    report = _report_with_sources(
        run_result.outcome.draft_report,
        run_result.outcome.sources,
    )
    if not report:
        run_result.output_error = "the paused checkpoint contains no draft report"
        return
    try:
        _atomic_write_text(run_result.output_path, report, overwrite=overwrite)
    except (OSError, ValueError) as exc:
        run_result.output_error = str(exc)
    else:
        run_result.output_written = True


# ── Run ────────────────────────────────────────────────────────────────────────────


async def _drive(args: argparse.Namespace) -> _CliRun:
    """Run one local CLI action against the durable SQLite checkpointer."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    data_dir = _data_dir(args)
    data_dir.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(data_dir / "checkpoints.sqlite")) as saver:
        await saver.setup()

        if args.export or args.export_draft:
            exporting_draft = bool(args.export_draft)
            session_id = _validate_session_id(args.export_draft or args.export)
            snapshot = await _checked_checkpoint(saver, session_id)
            outcome = (
                _draft_outcome_from_checkpoint(snapshot.values)
                if exporting_draft
                else _outcome_from_checkpoint(snapshot.values)
            )
            manifest = _load_manifest(data_dir, session_id) or {}
            output_path = _absolute_path(args.output)
            source = str(manifest.get("source") or "web")
            run_result = _CliRun(
                outcome=outcome,
                session_id=session_id,
                data_dir=data_dir,
                source=source,
                corpus_db=(
                    _absolute_path(manifest["corpus_db"])
                    if source == "corpus" and manifest.get("corpus_db")
                    else None
                ),
                output_path=output_path,
                event_log_path=(
                    _absolute_path(manifest["event_log_path"])
                    if manifest.get("event_log_path")
                    else None
                ),
            )
            if exporting_draft:
                _write_draft_output(run_result, overwrite=args.overwrite_output)
            else:
                _write_completed_output(run_result, overwrite=args.overwrite_output)
            if manifest and not exporting_draft:
                manifest["output_path"] = str(output_path)
                _write_json(_manifest_path(data_dir, session_id), manifest)
            return run_result

        is_new = args.query is not None
        raw_session_id = (
            args.session_id
            if is_new and args.session_id
            else f"local-{uuid.uuid4().hex[:12]}"
            if is_new
            else args.approve or args.reject or args.resume_plan
        )
        session_id = _validate_session_id(raw_session_id)

        if is_new:
            existing = await read_checkpoint(saver, session_id)
            if existing.outcome != CheckpointOutcome.MISSING:
                raise ValueError(
                    f"Session '{session_id}' already exists; choose another --session-id."
                )
            manifest = None
        else:
            snapshot = await _checked_checkpoint(saver, session_id)
            phase = _checkpoint_phase(snapshot.values)
            expected = "awaiting_plan" if args.resume_plan else "awaiting_approval"
            if phase != expected:
                raise ValueError(
                    f"Session '{session_id}' is {phase}, so this command cannot resume it; "
                    f"expected {expected}."
                )
            manifest = _load_manifest(data_dir, session_id)
            if manifest is None:
                print(
                    f"warning: session '{session_id}' predates CLI manifests; explicit "
                    "flags or current defaults will be used",
                    file=sys.stderr,
                )

        approval_chain = _approval_chain_from_manifest(manifest)
        pending_approval: dict[str, Any] | None = None

        load_env_file()
        options = _runtime_options(args, data_dir, manifest)
        run_config = run_config_from_env(fake=options.fake)
        saved_models = manifest.get("models") if manifest else None
        if isinstance(saved_models, Mapping) and dict(run_config.models) != dict(saved_models):
            raise ValueError(
                "The configured model routes differ from those recorded for this session. "
                "Restore the original MODEL_* environment values before resuming."
            )
        run_config = replace(
            run_config,
            corpus_mode=options.source == "corpus",
            skip_plan_gate=not options.plan_gate,
        )

        corpus_embedder = _embedder(options) if options.source == "corpus" else None
        if corpus_embedder is not None and not corpus_embedder.is_local:
            raise ValueError(
                "Corpus-only research guarantees zero network calls, but "
                f"'{options.embedding_base_url}' is not a local embedding endpoint."
            )
        corpus = (
            CorpusStore(options.corpus_db, corpus_embedder) if corpus_embedder is not None else None
        )
        if is_new and corpus is not None:
            await _require_searchable_corpus(corpus)

        with _EventReporter(
                session_id=session_id,
                quiet=args.quiet,
                verbose=args.verbose,
                path=options.event_log_path,
                new_session=is_new,
        ) as reporter:
            action = (
                "start"
                if is_new
                else "approve"
                if args.approve
                else "reject"
                if args.reject
                else "resume_plan"
            )
            reporter.lifecycle(
                "Starting research" if is_new else "Resuming research",
                action=action,
                source=options.source,
            )

            # Local credentials stay on RunConfig. The runner's separate provider-key port
            # is for a server user's BYOK override and must not enter the manifest or log.
            on_event = reporter if options.event_log_path is not None or not args.quiet else None
            ports = {
                "event_sink": InProcessEventSink(on_event=on_event),
                "cache": SqliteCache(data_dir / "cache.sqlite"),
                "corpus": corpus,
                "run_config": run_config,
            }

            if args.approve or args.reject:
                draft_report = snapshot.values.get("draft_report")
                if not isinstance(draft_report, str) or not draft_report:
                    raise ValueError("The session has no draft report to record at the review decision.")
                pending_approval = _approval_record(
                    action="approved" if args.approve else "rework_requested",
                    draft_hash=bundle.content_hash(draft_report),
                    feedback=args.feedback if args.reject else None,
                )
                outcome = await resume(
                    checkpointer=saver,
                    session_id=session_id,
                    approved=bool(args.approve),
                    feedback=args.feedback,
                    **ports,
                )
            elif args.resume_plan:
                plan = _load_plan(args.plan_file, session_id) if args.plan_file else {}
                pending_approval = _approval_record(
                    action="plan_approved",
                    draft_hash=bundle.content_hash(""),
                )
                outcome = await resume(
                    checkpointer=saver,
                    session_id=session_id,
                    plan=plan,
                    **ports,
                )
            else:
                outcome = await run(
                    checkpointer=saver,
                    session_id=session_id,
                    user_id="local",
                    query=args.query,
                    depth=args.depth,
                    **ports,
                )
                # Same-process approval still resumes the checkpoint; it never reruns Planner.
                if args.yes and outcome.status == "awaiting_approval":
                    if not outcome.draft_report:
                        raise ValueError("The session has no draft report to record at the review decision.")
                    pending_approval = _approval_record(
                        action="approved",
                        draft_hash=bundle.content_hash(outcome.draft_report),
                    )
                    outcome = await resume(
                        checkpointer=saver,
                        session_id=session_id,
                        approved=True,
                        **ports,
                    )

            plan_file = (
                _write_plan(data_dir, session_id, outcome)
                if outcome.status == "awaiting_plan"
                else None
            )
            if pending_approval is not None:
                approval_chain.append(pending_approval)
            _write_manifest(data_dir, session_id, options, run_config.models, approval_chain)

            run_result = _CliRun(
                outcome=outcome,
                session_id=session_id,
                data_dir=data_dir,
                source=options.source,
                corpus_db=options.corpus_db if options.source == "corpus" else None,
                plan_file=plan_file,
                output_path=options.output_path,
                event_log_path=options.event_log_path,
            )
            _write_completed_output(run_result, overwrite=args.overwrite_output)
            reporter.lifecycle(
                f"Research {outcome.status.replace('_', ' ')}",
                status=outcome.status,
                sources=len(outcome.sources),
                cost_usd=outcome.cost_usd,
                tokens_input=outcome.tokens_input,
                tokens_output=outcome.tokens_output,
                output_written=run_result.output_written,
                output_error=run_result.output_error,
            )
            return run_result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="research-engine",
        description="Run the research pipeline locally — no server, no Docker, no login.",
    )
    parser.add_argument("query", nargs="?", help="the research question")
    parser.add_argument(
        "--depth", default="balanced", choices=["fast", "balanced", "comprehensive"]
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        default=None,
        help="scripted models and fixture sources — no API key, no network",
    )
    parser.add_argument("--approve", metavar="SESSION_ID", help="finalize a paused draft")
    parser.add_argument("--reject", metavar="SESSION_ID", help="send a paused draft back")
    parser.add_argument("-f", "--feedback", help="feedback to accompany --reject")
    parser.add_argument("--yes", action="store_true", help="approve the report gate immediately")
    parser.add_argument("--session-id", help="use a specific session id instead of a random one")
    parser.add_argument(
        "--plan-gate",
        action="store_true",
        default=None,
        help="pause after planning so the proposed research design can be edited",
    )
    parser.add_argument("--resume-plan", metavar="SESSION_ID", help="approve or edit a paused plan")
    parser.add_argument("--plan-file", help="edited plan JSON for --resume-plan")
    parser.add_argument(
        "--source",
        choices=["web", "corpus"],
        help="evidence source for a new run (default: web)",
    )
    parser.add_argument("--corpus-db", help="corpus SQLite path (default: DATA_DIR/corpus.sqlite)")
    parser.add_argument(
        "--ingest",
        action="append",
        metavar="FILE",
        help="ingest a document; repeat the option to ingest more than one",
    )
    parser.add_argument("--list-corpus", action="store_true", help="list corpus documents")
    parser.add_argument("--corpus-status", action="store_true", help="show corpus status")
    parser.add_argument("--delete-document", metavar="DOCUMENT_ID", help="delete one document")
    parser.add_argument("--embedding-model", help="local Ollama embedding model")
    parser.add_argument("--embedding-base-url", help="local OpenAI-compatible base URL")
    parser.add_argument("--embedding-dimensions", type=int, help="embedding vector width")
    parser.add_argument("--output", help="write a completed report or exported draft to Markdown")
    parser.add_argument(
        "--overwrite-output", action="store_true", help="replace an existing output file"
    )
    parser.add_argument(
        "--export", metavar="SESSION_ID", help="export a completed checkpoint without rerunning"
    )
    parser.add_argument(
        "--export-draft",
        metavar="SESSION_ID",
        help="export a paused draft checkpoint without rerunning",
    )
    parser.add_argument(
        "--export-bundle",
        metavar="SESSION_ID",
        help="read and emit a verifier-approved bundle for a completed local session",
    )
    parser.add_argument(
        "--bundle-output",
        metavar="PATH",
        help="write --export-bundle JSON to this path instead of standard output",
    )
    parser.add_argument(
        "--verify-bundle",
        metavar="PATH",
        help="verify a .bundle.json file offline without reading local session state",
    )
    parser.add_argument(
        "--inspect",
        metavar="SESSION_ID",
        help="read evidence and sources from a checkpoint without rerunning",
    )
    parser.add_argument(
        "--show",
        choices=["evidence", "sources", "contradictions", "all"],
        help="checkpoint data selected by --inspect (default: all)",
    )
    parser.add_argument(
        "--data-dir", help=f"where to keep local state (default: {default_data_dir()})"
    )
    parser.add_argument("--json", action="store_true", help="emit the outcome as JSON")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show sanitized event details, including tool calls, on stderr",
    )
    parser.add_argument(
        "--event-log",
        metavar="PATH",
        help="persist sanitized pipeline events as JSON Lines; restored when resuming",
    )
    parser.add_argument("--quiet", action="store_true", help="don't stream progress to stderr")

    args = parser.parse_args(argv)
    corpus_actions = [
        bool(args.ingest),
        args.list_corpus,
        args.corpus_status,
        bool(args.delete_document),
    ]
    primary_actions = [
        args.query is not None,
        bool(args.approve),
        bool(args.reject),
        bool(args.resume_plan),
        bool(args.export),
        bool(args.export_draft),
        bool(args.export_bundle),
        bool(args.verify_bundle),
        bool(args.inspect),
        any(corpus_actions),
    ]
    if sum(primary_actions) != 1:
        parser.error(
            "choose exactly one action: QUERY, --approve, --reject, --resume-plan, "
            "--export, --export-draft, --export-bundle, --verify-bundle, --inspect, "
            "or one corpus-management action"
        )
    if sum(corpus_actions) > 1:
        parser.error(
            "choose only one of --ingest, --list-corpus, --corpus-status, or --delete-document"
        )
    if args.reject and not args.feedback:
        parser.error("--reject needs -f/--feedback saying what to fix")
    if args.feedback and not args.reject:
        parser.error("--feedback is valid only with --reject")
    if args.plan_file and not args.resume_plan:
        parser.error("--plan-file is valid only with --resume-plan")
    if args.plan_gate and args.query is None:
        parser.error("--plan-gate is valid only when starting a new query")
    if args.yes and args.query is None:
        parser.error("--yes is valid only when starting a new query")
    if args.yes and args.plan_gate:
        parser.error("--yes cannot cross the plan gate; review the plan explicitly")
    if args.session_id and args.query is None:
        parser.error("--session-id is valid only when starting a new query")
    if args.export and not args.output:
        parser.error("--export requires --output")
    if args.export_draft and not args.output:
        parser.error("--export-draft requires --output")
    if args.bundle_output and not args.export_bundle:
        parser.error("--bundle-output is valid only with --export-bundle")
    if args.export_bundle and args.output:
        parser.error("--export-bundle uses --bundle-output, not --output")
    if args.export_bundle and args.overwrite_output and not args.bundle_output:
        parser.error("--overwrite-output with --export-bundle requires --bundle-output")
    if args.show and not args.inspect:
        parser.error("--show is valid only with --inspect")
    if args.quiet and args.verbose:
        parser.error("--quiet and --verbose cannot be combined")
    run_action = bool(args.query is not None or args.approve or args.reject or args.resume_plan)
    if args.verbose and not run_action:
        parser.error("--verbose is valid only while running or resuming research")
    if args.event_log and not run_action:
        parser.error("--event-log is valid only while running or resuming research")
    if args.inspect and any(
            (
                    args.fake is not None,
                    args.source is not None,
                    args.corpus_db is not None,
                    args.embedding_model is not None,
                    args.embedding_base_url is not None,
                    args.embedding_dimensions is not None,
                    args.output is not None,
                    args.overwrite_output,
                    args.quiet,
            )
    ):
        parser.error("--inspect accepts only --show, --json, and --data-dir")
    if args.export_bundle and any(
            value is not None
            for value in (
                args.fake,
                args.source,
                args.corpus_db,
                args.embedding_model,
                args.embedding_base_url,
                args.embedding_dimensions,
                args.event_log,
            )
    ):
        parser.error("--export-bundle accepts only --bundle-output, --overwrite-output, and --data-dir")
    if args.export_bundle and args.json:
        parser.error("--export-bundle already emits JSON; do not combine it with --json")
    if args.verify_bundle and any(
            (
                args.fake is not None,
                args.source is not None,
                args.corpus_db is not None,
                args.embedding_model is not None,
                args.embedding_base_url is not None,
                args.embedding_dimensions is not None,
                args.output is not None,
                args.bundle_output is not None,
                args.overwrite_output,
                args.event_log is not None,
                args.quiet,
                args.verbose,
                args.data_dir is not None,
            )
    ):
        parser.error("--verify-bundle accepts only --json")
    if any(corpus_actions) and (args.output or args.overwrite_output):
        parser.error("report output options cannot be combined with corpus management")
    if args.export and any(
            value is not None
            for value in (
                    args.fake,
                    args.source,
                    args.corpus_db,
                    args.embedding_model,
                    args.embedding_base_url,
                    args.embedding_dimensions,
            )
    ):
        parser.error("--export accepts no run or source configuration options")
    if args.export_draft and any(
            value is not None
            for value in (
                    args.fake,
                    args.source,
                    args.corpus_db,
                    args.embedding_model,
                    args.embedding_base_url,
                    args.embedding_dimensions,
            )
    ):
        parser.error("--export-draft accepts no run or source configuration options")
    if args.overwrite_output and not args.output and args.query is not None:
        parser.error("--overwrite-output needs --output for a new query")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.verify_bundle:
            verification = verify_bundle.verify_file(args.verify_bundle)
            print(
                verify_bundle.format_json(verification)
                if args.json
                else verify_bundle.format_text(verification)
            )
            return 0 if verification.passed else 1

        if args.export_bundle:
            serialized_bundle = asyncio.run(_export_bundle(args))
            if args.bundle_output:
                output_path = _absolute_path(args.bundle_output)
                _atomic_write_text(
                    output_path,
                    serialized_bundle,
                    overwrite=args.overwrite_output,
                )
                print(f"Bundle written to: {output_path}")
            else:
                print(serialized_bundle, end="")
            return 0

        if any((args.ingest, args.list_corpus, args.corpus_status, args.delete_document)):
            result = asyncio.run(_manage_corpus(args))
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                _print_corpus_result(result)
            return 0

        if args.inspect:
            result = asyncio.run(_inspect_checkpoint(args))
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            else:
                _print_inspection(result)
            return 0

        run_result = asyncio.run(_drive(args))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(_run_payload(run_result), ensure_ascii=False, indent=2))
    else:
        _print_outcome(run_result)

    if run_result.output_error:
        return 2
    return 1 if run_result.outcome.status == "failed" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
