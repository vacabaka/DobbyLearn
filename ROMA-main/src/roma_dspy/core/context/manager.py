"""
Context Manager for building execution context for ROMA-DSPy agents.

The ContextManager is responsible for:
1. Building Pydantic context models from runtime state
2. Composing fundamental + agent-specific context
3. Serializing to XML strings for DSPy signatures

It follows the Single Responsibility Principle: one job is context orchestration.
"""

from datetime import datetime, UTC
from typing import List, Optional, TYPE_CHECKING
from roma_dspy.core.context.models import (
    TemporalContext,
    FileSystemContext,
    RecursionContext,
    ToolsContext,
    ToolInfo,
    FundamentalContext,
    ExecutorSpecificContext,
    PlannerSpecificContext,
    DependencyResult,
    ParentResult,
    SiblingResult,
)
from roma_dspy.types import TaskStatus

if TYPE_CHECKING:
    from ..engine.runtime import ModuleRuntime
    from ..engine.dag import TaskDAG
    from ..signatures.base_models.task_node import TaskNode
    from ..storage import FileStorage


class ContextManager:
    """
    Central context manager that orchestrates context building for all agents.

    Design principles:
    - Uses Pydantic models for type safety and validation
    - Separates data (models) from building logic (this class)
    - Returns XML strings ready for DSPy signatures
    - Follows DRY: shared components built once, composed differently per agent

    Usage:
        manager = ContextManager(file_storage, overall_objective)
        context_xml = manager.build_executor_context(task, tools_data, runtime, dag)
        # Pass context_xml to executor signature
    """

    def __init__(self, file_storage: "FileStorage", overall_objective: str):
        """
        Initialize context manager.

        Args:
            file_storage: FileStorage instance for this execution (provides paths and execution_id)
            overall_objective: Root goal of execution (helps agents align with bigger picture)
        """
        self.file_storage = file_storage
        self.overall_objective = overall_objective

    # ==================== Component Builders (Private) ====================

    def _build_temporal(self) -> TemporalContext:
        """Build temporal context with current date/time."""
        now = datetime.now(UTC)
        return TemporalContext(
            current_date=now.strftime("%Y-%m-%d"),
            current_year=now.year,
            current_timestamp=now.isoformat()
        )

    def _build_file_system(self) -> FileSystemContext:
        """Build file system context from FileStorage instance."""
        return FileSystemContext.from_file_storage(self.file_storage)

    def _build_recursion(self, task: "TaskNode") -> RecursionContext:
        """Build recursion context from task's depth information."""
        return RecursionContext(
            current_depth=task.depth,
            max_depth=task.max_depth,
            at_limit=task.depth >= task.max_depth
        )

    def _build_tools(self, tools_data: List[dict]) -> ToolsContext:
        """Build tools context from tools data."""
        tools = [ToolInfo(name=t["name"], description=t["description"]) for t in tools_data]
        return ToolsContext(tools=tools)

    def _build_fundamental(
        self,
        task: "TaskNode",
        tools_data: List[dict],
        include_file_system: bool = False
    ) -> FundamentalContext:
        """Build fundamental context available to all agents."""
        return FundamentalContext(
            overall_objective=self.overall_objective,
            temporal=self._build_temporal(),
            recursion=self._build_recursion(task),
            tools=self._build_tools(tools_data),
            file_system=self._build_file_system() if include_file_system else None
        )

    def _build_executor_specific(
        self,
        task: "TaskNode",
        runtime: "ModuleRuntime",
        dag: "TaskDAG"
    ) -> ExecutorSpecificContext:
        """Build executor-specific context with dependency results."""
        dependency_results = []

        if task.dependencies:
            for dep_id in task.dependencies:
                result_str = runtime.context_store.get_result(dep_id)
                if result_str:
                    try:
                        dep_task, _ = dag.find_node(dep_id)
                        dependency_results.append(
                            DependencyResult(goal=dep_task.goal, output=result_str)
                        )
                    except ValueError:
                        pass  # Dependency not found in DAG

        return ExecutorSpecificContext(dependency_results=dependency_results)

    def _build_planner_specific(
        self,
        task: "TaskNode",
        runtime: "ModuleRuntime",
        dag: "TaskDAG"
    ) -> PlannerSpecificContext:
        """Build planner-specific context with parent and sibling results."""
        parent_results = []
        sibling_results = []

        # Get parent result
        if task.parent_id:
            parent_result = runtime.context_store.get_result(task.parent_id)
            if parent_result:
                # BUG FIX: Use find_node for hierarchical lookup (parent is in parent DAG, not subgraph)
                try:
                    parent_task, _ = dag.find_node(task.parent_id)
                    parent_results.append(ParentResult(goal=parent_task.goal, result=parent_result))
                except ValueError:
                    from loguru import logger
                    logger.warning(
                        f"[build_planner_context] Parent task {task.parent_id[:8]}... not found in DAG hierarchy"
                    )

        # Get sibling results
        if task.parent_id:
            # BUG FIX: Use find_node for hierarchical lookup (parent is in parent DAG, not subgraph)
            try:
                parent, _ = dag.find_node(task.parent_id)
            except ValueError:
                from loguru import logger
                logger.warning(
                    f"[build_planner_context] Parent task {task.parent_id[:8]}... not found for sibling lookup"
                )
                parent = None
            if parent and parent.subgraph_id:
                subgraph = dag.get_subgraph(parent.subgraph_id)
                for sibling in subgraph.get_all_tasks(include_subgraphs=False):
                    if sibling.task_id != task.task_id and sibling.status == TaskStatus.COMPLETED:
                        sib_result = runtime.context_store.get_result(sibling.task_id)
                        if sib_result:
                            sibling_results.append(SiblingResult(goal=sibling.goal, result=sib_result))

        return PlannerSpecificContext(parent_results=parent_results, sibling_results=sibling_results)

    # ==================== Generic Builder (DRY) ====================

    def _build_context(
        self,
        task: "TaskNode",
        tools_data: List[dict],
        include_file_system: bool = False,
        specific_context: Optional[str] = None
    ) -> str:
        """
        Generic context builder - composes fundamental + agent-specific context.

        Args:
            task: Current task node
            tools_data: Available tools information
            include_file_system: Whether to include file system in fundamental context
            specific_context: Optional agent-specific context XML (or None for agents with no specific context)

        Returns:
            Complete XML context string
        """
        fundamental = self._build_fundamental(task, tools_data, include_file_system)

        parts = ["<context>", fundamental.to_xml()]
        if specific_context:
            parts.append(specific_context)
        parts.append("</context>")

        return '\n'.join(parts)

    # ==================== Public API: Agent-Specific Builders ====================

    def build_atomizer_context(self, task: "TaskNode", tools_data: List[dict]) -> str:
        """Build complete context for Atomizer agent (fundamental only)."""
        return self._build_context(task, tools_data, include_file_system=False)

    def build_planner_context(
        self,
        task: "TaskNode",
        tools_data: List[dict],
        runtime: "ModuleRuntime",
        dag: "TaskDAG"
    ) -> str:
        """Build complete context for Planner agent (fundamental + parent/siblings)."""
        specific = self._build_planner_specific(task, runtime, dag)
        return self._build_context(task, tools_data, include_file_system=False, specific_context=specific.to_xml())

    def build_executor_context(
        self,
        task: "TaskNode",
        tools_data: List[dict],
        runtime: "ModuleRuntime",
        dag: "TaskDAG"
    ) -> str:
        """Build complete context for Executor agent (fundamental + file_system + dependencies)."""
        specific = self._build_executor_specific(task, runtime, dag)
        return self._build_context(task, tools_data, include_file_system=True, specific_context=specific.to_xml())

    def build_aggregator_context(self, task: "TaskNode", tools_data: List[dict]) -> str:
        """Build complete context for Aggregator agent (fundamental + note about child results)."""
        note = "<aggregator_specific><note>Child results are provided in subtasks_results field</note></aggregator_specific>"
        return self._build_context(task, tools_data, include_file_system=False, specific_context=note)

    def build_verifier_context(self, task: "TaskNode", tools_data: List[dict]) -> str:
        """Build complete context for Verifier agent (fundamental only)."""
        return self._build_context(task, tools_data, include_file_system=False)
