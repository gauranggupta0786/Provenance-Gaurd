"""Week 4 typed LangGraph lifecycle with durable, inspectable checkpoints.

The graph coordinates work; it does not replace ProvenanceGuard's signed
records or atomic SQLite gateway.  The execute callback must enter through
that gateway, which remains the sole authority for protected writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable

from langgraph.graph import END, START, StateGraph

from .trusted_models import (
    AgentRunState,
    EvaluationRecord,
    ExecutionRecord,
    ModelCallRecord,
    NodeDiagnostic,
    PolicyDecision,
    PolicyOutcome,
    ReceiptRecord,
    RunStatus,
    SourceEvidence,
    ToolClass,
    ToolIntent,
    WorkflowNode,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class LifecycleCheckpointStore:
    """Durable Pydantic-state checkpoints and append-only node diagnostics."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.RLock()
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS graph_checkpoints (
                    graph_run_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    current_node TEXT NOT NULL,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS graph_node_records (
                    graph_run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    node TEXT NOT NULL,
                    status TEXT NOT NULL,
                    diagnostic_json TEXT NOT NULL,
                    PRIMARY KEY (graph_run_id, sequence)
                );
                CREATE TRIGGER IF NOT EXISTS graph_nodes_no_update
                BEFORE UPDATE ON graph_node_records BEGIN SELECT RAISE(ABORT, 'append-only'); END;
                CREATE TRIGGER IF NOT EXISTS graph_nodes_no_delete
                BEFORE DELETE ON graph_node_records BEGIN SELECT RAISE(ABORT, 'append-only'); END;
            """)

    def save(self, state: AgentRunState, diagnostic: NodeDiagnostic) -> None:
        # Reparse serialized content before it crosses into SQLite.
        checked = AgentRunState.model_validate_json(state.model_dump_json())
        checked_diagnostic = NodeDiagnostic.model_validate_json(diagnostic.model_dump_json())
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO graph_node_records VALUES (?,?,?,?,?)",
                (
                    checked.graph_run_id,
                    checked_diagnostic.sequence,
                    checked_diagnostic.node.value,
                    checked_diagnostic.status,
                    checked_diagnostic.model_dump_json(),
                ),
            )
            self._conn.execute(
                "INSERT INTO graph_checkpoints VALUES (?,?,?,?,?) "
                "ON CONFLICT(graph_run_id) DO UPDATE SET "
                "state_json=excluded.state_json,current_node=excluded.current_node," 
                "status=excluded.status,updated_at=excluded.updated_at",
                (
                    checked.graph_run_id,
                    checked.model_dump_json(),
                    checked.current_node.value,
                    checked.status.value,
                    _now(),
                ),
            )

    def load(self, graph_run_id: str) -> AgentRunState | None:
        row = self._conn.execute(
            "SELECT state_json FROM graph_checkpoints WHERE graph_run_id=?", (graph_run_id,),
        ).fetchone()
        return AgentRunState.model_validate_json(row[0]) if row else None

    def diagnostics(self, graph_run_id: str) -> list[NodeDiagnostic]:
        rows = self._conn.execute(
            "SELECT diagnostic_json FROM graph_node_records WHERE graph_run_id=? ORDER BY sequence",
            (graph_run_id,),
        ).fetchall()
        return [NodeDiagnostic.model_validate_json(row[0]) for row in rows]

    def close(self) -> None:
        self._conn.close()


@dataclass
class LifecycleServices:
    call_model: Callable[[AgentRunState], ModelCallRecord]
    parse_tool_intent: Callable[[AgentRunState], ToolIntent]
    label_sources: Callable[[AgentRunState], tuple[SourceEvidence, ...]]
    evaluate_policy: Callable[[AgentRunState], PolicyOutcome]
    build_proposal: Callable[[AgentRunState], str]
    obtain_approval: Callable[[AgentRunState], str | None]
    execute_read: Callable[[AgentRunState], None]
    execute_write: Callable[[AgentRunState], tuple[ExecutionRecord, ReceiptRecord]]
    evaluate_final: Callable[[AgentRunState], EvaluationRecord]
    read_tools: frozenset[str] = frozenset()


class LifecycleGraph:
    """Compiled graph whose every node consumes and emits validated state."""

    def __init__(self, services: LifecycleServices, checkpoints: LifecycleCheckpointStore):
        self.services = services
        self.checkpoints = checkpoints
        self._next = {
            WorkflowNode.LOAD_SCENARIO: WorkflowNode.BUILD_CONVERSATION,
            WorkflowNode.BUILD_CONVERSATION: WorkflowNode.CALL_MODEL,
            WorkflowNode.CALL_MODEL: WorkflowNode.PARSE_TOOL_INTENT,
            WorkflowNode.PARSE_TOOL_INTENT: WorkflowNode.CLASSIFY_TOOL,
            WorkflowNode.EXECUTE_READ: WorkflowNode.CALL_MODEL,
            WorkflowNode.LABEL_SOURCES: WorkflowNode.VALIDATE_WRITE,
            WorkflowNode.VALIDATE_WRITE: WorkflowNode.EVALUATE_POLICY,
            WorkflowNode.REJECT_AND_AUDIT: WorkflowNode.EVALUATE_FINAL_STATE,
            WorkflowNode.BUILD_AND_SIGN_PROPOSAL: WorkflowNode.OBTAIN_APPROVAL,
            WorkflowNode.OBTAIN_APPROVAL: WorkflowNode.GATEWAY_REVALIDATE,
            WorkflowNode.GATEWAY_REVALIDATE: WorkflowNode.CONSUME_AND_RESERVE,
            WorkflowNode.CONSUME_AND_RESERVE: WorkflowNode.EXECUTE_ADAPTER,
            WorkflowNode.EXECUTE_ADAPTER: WorkflowNode.PERSIST_RECEIPT,
            WorkflowNode.PERSIST_RECEIPT: WorkflowNode.EVALUATE_FINAL_STATE,
        }
        self.compiled = self._compile()

    def _transition(
        self,
        state: AgentRunState,
        node: WorkflowNode,
        *,
        status: RunStatus = RunStatus.RUNNING,
        error: Exception | str | None = None,
        started: float,
        **updates,
    ) -> AgentRunState:
        error_text = str(error) if error is not None else None
        diagnostic_status = (
            "INTERRUPTED" if status is RunStatus.INTERRUPTED else
            "FAILED" if status in {RunStatus.FAILED, RunStatus.RETRYABLE} else "SUCCEEDED"
        )
        diagnostic = NodeDiagnostic(
            graph_run_id=state.graph_run_id,
            node=node,
            sequence=len(state.node_diagnostics) + 1,
            started_at=_now(),
            latency_ms=max(0, int((time.monotonic() - started) * 1000)),
            status=diagnostic_status,
            evidence_ref=f"sqlite://graph_node_records/{state.graph_run_id}/{len(state.node_diagnostics) + 1}",
            error_type=type(error).__name__ if isinstance(error, Exception) else ("Interruption" if error else None),
            error=error_text,
        )
        checked = state.revalidated(
            current_node=node,
            status=status,
            last_error=error_text,
            path=(*state.path, node),
            node_diagnostics=(*state.node_diagnostics, diagnostic),
            **updates,
        )
        self.checkpoints.save(checked, diagnostic)
        return checked

    @staticmethod
    def _dump(state: AgentRunState) -> dict:
        return state.model_dump(mode="python")

    def _simple(self, node: WorkflowNode, action: Callable[[AgentRunState], dict] | None = None):
        def run(state: AgentRunState) -> dict:
            started = time.monotonic()
            try:
                updates = action(state) if action is not None else {}
                return self._dump(self._transition(state, node, started=started, **updates))
            except Exception as exc:
                if node is WorkflowNode.CALL_MODEL and isinstance(exc, TimeoutError) and state.retry_count == 0:
                    return self._dump(self._transition(
                        state, node, status=RunStatus.RETRYABLE, error=exc,
                        retry_count=1, started=started,
                    ))
                return self._dump(self._transition(
                    state, node, status=RunStatus.FAILED, error=exc, started=started,
                ))
        return run

    def _approval(self, state: AgentRunState) -> dict:
        started = time.monotonic()
        try:
            approval_id = self.services.obtain_approval(state)
            if approval_id is None:
                return self._dump(self._transition(
                    state, WorkflowNode.OBTAIN_APPROVAL,
                    status=RunStatus.INTERRUPTED,
                    error="approval decision is required",
                    started=started,
                ))
            return self._dump(self._transition(
                state, WorkflowNode.OBTAIN_APPROVAL,
                approval_id=approval_id, started=started,
            ))
        except Exception as exc:
            return self._dump(self._transition(
                state, WorkflowNode.OBTAIN_APPROVAL,
                status=RunStatus.FAILED, error=exc, started=started,
            ))

    def _evaluate_final(self, state: AgentRunState) -> dict:
        started = time.monotonic()
        try:
            evaluation = self.services.evaluate_final(state)
            evaluated = self._transition(
                state, WorkflowNode.EVALUATE_FINAL_STATE,
                evaluation=evaluation, started=started,
            )
            return self._dump(self._transition(
                evaluated, WorkflowNode.END, status=RunStatus.SUCCEEDED,
                started=time.monotonic(),
            ))
        except Exception as exc:
            return self._dump(self._transition(
                state, WorkflowNode.EVALUATE_FINAL_STATE,
                status=RunStatus.FAILED, error=exc, started=started,
            ))

    def _route_next(self, next_node: WorkflowNode):
        def route(state: AgentRunState) -> str:
            if state.status is RunStatus.RETRYABLE:
                return WorkflowNode.CALL_MODEL.value
            if state.status in {RunStatus.FAILED, RunStatus.INTERRUPTED}:
                return "__end__"
            return next_node.value
        return route

    def _entry(self, state: AgentRunState) -> str:
        if state.status in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.REJECTED}:
            return "__end__"
        if state.status in {RunStatus.INTERRUPTED, RunStatus.RETRYABLE}:
            return state.current_node.value
        if state.current_node is WorkflowNode.START:
            return WorkflowNode.LOAD_SCENARIO.value
        return self._next.get(state.current_node, WorkflowNode.EVALUATE_FINAL_STATE).value

    def _compile(self):
        graph = StateGraph(AgentRunState)
        actions = {
            WorkflowNode.LOAD_SCENARIO: lambda s: {},
            WorkflowNode.BUILD_CONVERSATION: lambda s: {
                "conversation_evidence_ref": s.conversation_evidence_ref or f"evidence://{s.graph_run_id}/conversation"
            },
            WorkflowNode.CALL_MODEL: lambda s: {"model_call": self.services.call_model(s)},
            WorkflowNode.PARSE_TOOL_INTENT: lambda s: {"tool_intent": self.services.parse_tool_intent(s)},
            WorkflowNode.CLASSIFY_TOOL: lambda s: {
                "tool_class": ToolClass.READ if s.tool_intent and s.tool_intent.tool_name in self.services.read_tools else ToolClass.WRITE
            },
            WorkflowNode.EXECUTE_READ: lambda s: (self.services.execute_read(s), {})[1],
            WorkflowNode.LABEL_SOURCES: lambda s: {"source_evidence": self.services.label_sources(s)},
            WorkflowNode.VALIDATE_WRITE: lambda s: {} if s.tool_intent and s.tool_intent.valid else (_ for _ in ()).throw(ValueError("invalid write intent")),
            WorkflowNode.EVALUATE_POLICY: lambda s: {"policy_outcome": self.services.evaluate_policy(s)},
            WorkflowNode.REJECT_AND_AUDIT: lambda s: {},
            WorkflowNode.BUILD_AND_SIGN_PROPOSAL: lambda s: {"proposal_id": self.services.build_proposal(s)},
            WorkflowNode.GATEWAY_REVALIDATE: lambda s: {},
            WorkflowNode.CONSUME_AND_RESERVE: lambda s: {},
            WorkflowNode.EXECUTE_ADAPTER: lambda s: dict(zip(("execution", "receipt"), self.services.execute_write(s))),
            WorkflowNode.PERSIST_RECEIPT: lambda s: {},
        }
        for node, action in actions.items():
            graph.add_node(node.value, self._simple(node, action))
        graph.add_node(WorkflowNode.OBTAIN_APPROVAL.value, self._approval)
        graph.add_node(WorkflowNode.EVALUATE_FINAL_STATE.value, self._evaluate_final)

        entry_names = [node.value for node in WorkflowNode if node not in {WorkflowNode.START, WorkflowNode.END}]
        graph.add_conditional_edges(START, self._entry, {**{name: name for name in entry_names}, "__end__": END})

        for node, next_node in self._next.items():
            if node in {WorkflowNode.OBTAIN_APPROVAL}:
                continue
            graph.add_conditional_edges(
                node.value,
                self._route_next(next_node),
                {next_node.value: next_node.value, WorkflowNode.CALL_MODEL.value: WorkflowNode.CALL_MODEL.value, "__end__": END},
            )
        graph.add_conditional_edges(
            WorkflowNode.CLASSIFY_TOOL.value,
            lambda s: "__end__" if s.status is RunStatus.FAILED else (
                WorkflowNode.EXECUTE_READ.value if s.tool_class is ToolClass.READ else WorkflowNode.LABEL_SOURCES.value
            ),
            {WorkflowNode.EXECUTE_READ.value: WorkflowNode.EXECUTE_READ.value,
             WorkflowNode.LABEL_SOURCES.value: WorkflowNode.LABEL_SOURCES.value, "__end__": END},
        )
        graph.add_conditional_edges(
            WorkflowNode.EVALUATE_POLICY.value,
            lambda s: "__end__" if s.status is RunStatus.FAILED else (
                WorkflowNode.REJECT_AND_AUDIT.value if s.policy_outcome and s.policy_outcome.decision is PolicyDecision.DENY
                else WorkflowNode.BUILD_AND_SIGN_PROPOSAL.value
            ),
            {WorkflowNode.REJECT_AND_AUDIT.value: WorkflowNode.REJECT_AND_AUDIT.value,
             WorkflowNode.BUILD_AND_SIGN_PROPOSAL.value: WorkflowNode.BUILD_AND_SIGN_PROPOSAL.value,
             "__end__": END},
        )
        graph.add_conditional_edges(
            WorkflowNode.OBTAIN_APPROVAL.value,
            self._route_next(WorkflowNode.GATEWAY_REVALIDATE),
            {WorkflowNode.GATEWAY_REVALIDATE.value: WorkflowNode.GATEWAY_REVALIDATE.value,
             WorkflowNode.CALL_MODEL.value: WorkflowNode.CALL_MODEL.value, "__end__": END},
        )
        graph.add_edge(WorkflowNode.EVALUATE_FINAL_STATE.value, END)
        return graph.compile()

    def invoke(self, state: AgentRunState) -> AgentRunState:
        checked = AgentRunState.model_validate(state)
        result = self.compiled.invoke(checked)
        return AgentRunState.model_validate(result)

    def resume(self, graph_run_id: str) -> AgentRunState:
        state = self.checkpoints.load(graph_run_id)
        if state is None:
            raise KeyError(f"no checkpoint for graph run {graph_run_id!r}")
        return self.invoke(state)

    def mermaid(self) -> str:
        return self.compiled.get_graph().draw_mermaid()
