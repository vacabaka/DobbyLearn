# Prompt Optimization

Modular toolkit for optimizing ROMA-DSPy prompts using GEPA (Generative Expectation-Maximization Prompt Optimization with Adversarial Examples).

## Structure

```
prompt_optimization/
├── config.py              # Configuration management (dataclasses)
├── datasets.py            # Dataset loaders (AIMO, AIME)
├── solver_setup.py        # Solver factory with instruction constants
├── judge.py              # LLM judge for component evaluation
├── metrics/              # Metric implementations (basic, search, number, feedback)
│   ├── __init__.py
│   ├── base.py
│   ├── metric_with_feedback.py
│   ├── number_metric.py
│   └── search_metric.py
├── selectors.py          # Component selector strategies
├── optimizer.py          # GEPA optimizer factory
└── run_optimization.py   # Main CLI script
```

## Quick Start

### Basic Usage

```bash
# Run with defaults (5 train, 5 val, 15 test)
python -m prompt_optimization.run_optimization

# Customize dataset sizes
python -m prompt_optimization.run_optimization --train-size 10 --val-size 10 --test-size 30

# Use different component selector
python -m prompt_optimization.run_optimization --selector round_robin

# Save optimized program
python -m prompt_optimization.run_optimization --output optimized_solver.json

# Enable verbose logging
python -m prompt_optimization.run_optimization --verbose
```

### Available Selectors

- `planner_only` (default) - Optimize only the planner component
- `atomizer_only` - Optimize only the atomizer component
- `executor_only` - Optimize only the executor component
- `aggregator_only` - Optimize only the aggregator component
- `round_robin` - Cycle through all components

### Programmatic Usage

```python
from prompt_optimization import (
    get_default_config,
    load_aimo_datasets,
    create_solver_module,
    ComponentJudge,
    MetricWithFeedback,
    create_optimizer
)

# Load config
config = get_default_config()
config.train_size = 10
config.max_metric_calls = 20

# Load datasets
train, val, test = load_aimo_datasets(
    train_size=config.train_size,
    val_size=config.val_size,
    test_size=config.test_size
)

# Create solver
solver = create_solver_module(config)

# Create judge and metric
judge = ComponentJudge(config.judge_lm)
# Wrap a scoring metric (defaults to basic integer comparison if omitted)
metric = MetricWithFeedback(judge)

# Create optimizer
optimizer = create_optimizer(config, metric, component_selector="planner_only")

# Alternative: plug in a custom scoring metric (e.g., search accuracy)
# from prompt_optimization.metrics import NumberMetric
# metric = MetricWithFeedback(judge, scoring_metric=NumberMetric())

# Run optimization
optimized = optimizer.compile(solver, trainset=train, valset=val)
```

### Async Usage

Both the judge and metrics support async execution for improved performance:

```python
import asyncio
from prompt_optimization import ComponentJudge, MetricWithFeedback, get_default_config

config = get_default_config()

# Create judge
judge = ComponentJudge(config.judge_lm)

# Sync usage
feedback = judge(
    component_name="planner",
    component_trace={"subtasks": [...], "dependencies_graph": {...}},
    prediction_trace="Full trace..."
)

# Async usage
async def evaluate_async():
    feedback = await judge.__acall__(
        component_name="planner",
        component_trace={"subtasks": [...], "dependencies_graph": {...}},
        prediction_trace="Full trace..."
    )
    return feedback

# Run async
feedback = asyncio.run(evaluate_async())

# Metrics also support async
metric = MetricWithFeedback(judge)

# Async metric evaluation
async def evaluate_metric():
    result = await metric.aforward(
        example=example,
        prediction=prediction,
        pred_name="planner",
        pred_trace=trace
    )
    return result
```

## Configuration

All configuration is centralized in `config.py`. Key parameters:

- **LM Configs**: Model names, temperatures, max tokens for each component
- **Dataset**: Train/val/test sizes, random seed
- **Execution**: Max parallel executions, concurrency limits
- **GEPA**: Max metric calls, num threads, reflection minibatch size
- **Solver**: Max depth, logging settings

## CLI Options

```
usage: run_optimization.py [-h] [--train-size TRAIN_SIZE] [--val-size VAL_SIZE]
                           [--test-size TEST_SIZE] [--max-parallel MAX_PARALLEL]
                           [--concurrency CONCURRENCY] [--max-metric-calls MAX_METRIC_CALLS]
                           [--num-threads NUM_THREADS]
                           [--selector {planner_only,atomizer_only,executor_only,aggregator_only,round_robin}]
                           [--output OUTPUT] [--verbose] [--skip-eval]
```

## Key Features

- ✅ **No global state** - All components properly initialized and passed as dependencies
- ✅ **Uses AsyncParallelExecutor** - Leverages existing async utilities from `roma_dspy.utils`
- ✅ **Async support** - Judge and metrics support both sync (`forward`) and async (`aforward`) execution
- ✅ **Fully configurable** - Dataclass-based configuration with CLI overrides
- ✅ **Modular** - Each component is independently testable and reusable
- ✅ **Type-safe** - Proper typing throughout
- ✅ **Clean CLI** - Runnable as `python -m prompt_optimization.run_optimization`

## Migration from Notebook

The refactoring extracts all logic from `prompt_optimization.ipynb`:

| Notebook Cell | New Location |
|---------------|--------------|
| LM configs | `config.py:LMConfig` |
| Module initialization | `solver_setup.py:create_solver_module()` |
| Instructions | `solver_setup.py` (constants) |
| Dataset loading | `datasets.py:load_aimo_datasets()` |
| Judge setup | `judge.py:ComponentJudge` |
| Metrics | `metrics/__init__.py:basic_metric`, `metrics/metric_with_feedback.py:MetricWithFeedback` |
| Selectors | `selectors.py:*_selector` |
| GEPA optimizer | `optimizer.py:create_optimizer()` |
| Execution | `run_optimization.py:main()` |
