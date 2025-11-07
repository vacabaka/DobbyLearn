#!/usr/bin/env python3
"""Example demonstrating async usage of ComponentJudge and MetricWithFeedback."""

import asyncio
import dspy
from prompt_optimization import (
    ComponentJudge,
    MetricWithFeedback,
    get_default_config,
    load_aimo_datasets,
    create_solver_module,
)
from prompt_optimization.prompts import GRADER_PROMPT


async def example_async_judge():
    """Demonstrate async judge usage."""
    print("=" * 60)
    print("Async Judge Example")
    print("=" * 60)

    # Setup config
    config = get_default_config()

    # Create judge
    judge = ComponentJudge(GRADER_PROMPT, config.judge_lm)

    # Example component trace
    component_trace = {
        "subtasks": [
            {"id": "1", "goal": "Parse the problem", "task_type": "THINK"},
            {"id": "2", "goal": "Calculate result", "task_type": "THINK"},
        ],
        "dependencies_graph": {"1": [], "2": ["1"]},
    }

    # Async evaluation
    print("\nEvaluating component asynchronously...")
    feedback = await judge.__acall__(
        component_name="planner",
        component_trace=component_trace,
        prediction_trace="Sample execution trace...",
    )

    print(f"\nFeedback received (length: {len(feedback)} chars)")
    print(f"First 200 chars: {feedback[:200]}...")


async def example_async_metric():
    """Demonstrate async metric usage with parallel evaluations."""
    print("\n" + "=" * 60)
    print("Async Metric Example (Parallel Evaluations)")
    print("=" * 60)

    # Setup
    config = get_default_config()
    judge = ComponentJudge(GRADER_PROMPT, config.judge_lm)
    metric = MetricWithFeedback(judge)

    # Create mock examples
    examples = [
        dspy.Example({"goal": f"Problem {i}", "answer": str(i)}).with_inputs("goal")
        for i in range(3)
    ]

    predictions = [
        dspy.Prediction(result_text=str(i)) for i in range(3)
    ]

    # Evaluate in parallel using async
    print(f"\nEvaluating {len(examples)} examples in parallel...")

    async def eval_one(example, prediction, idx):
        result = await metric.aforward(
            example=example,
            prediction=prediction,
            pred_name="planner",
            pred_trace={"subtask_count": idx + 1},
        )
        return result

    # Run all evaluations concurrently
    results = await asyncio.gather(
        *[eval_one(ex, pred, i) for i, (ex, pred) in enumerate(zip(examples, predictions))]
    )

    print(f"\nCompleted {len(results)} evaluations")
    for i, result in enumerate(results):
        print(f"  Example {i}: score={result.score}, feedback_length={len(result.feedback)}")


async def example_batch_async():
    """Demonstrate batch async evaluation."""
    print("\n" + "=" * 60)
    print("Batch Async Evaluation Example")
    print("=" * 60)

    config = get_default_config()
    judge = ComponentJudge(GRADER_PROMPT, config.judge_lm)

    # Simulate multiple component traces
    traces = [
        {"component": "planner", "subtasks": i + 1}
        for i in range(5)
    ]

    print(f"\nEvaluating {len(traces)} components in parallel...")

    # Batch async evaluation
    feedbacks = await asyncio.gather(
        *[
            judge.__acall__(
                component_name=f"component_{i}",
                component_trace=trace,
            )
            for i, trace in enumerate(traces)
        ]
    )

    print(f"\nReceived {len(feedbacks)} feedback responses")
    avg_length = sum(len(f) for f in feedbacks) / len(feedbacks)
    print(f"Average feedback length: {avg_length:.0f} chars")


async def main():
    """Run all async examples."""
    print("\n🚀 Starting Async Examples\n")

    # Run examples
    await example_async_judge()
    await example_async_metric()
    await example_batch_async()

    print("\n✅ All async examples completed!\n")


if __name__ == "__main__":
    asyncio.run(main())
