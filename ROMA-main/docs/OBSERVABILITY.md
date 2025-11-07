# Observability and Monitoring

ROMA-DSPy provides comprehensive observability through MLflow integration, enabling experiment tracking, metrics logging, and execution tracing.

## Overview

The observability system captures:
- **Execution traces** - Task decomposition and execution flow
- **LLM metrics** - Token usage, costs, and latency for each LLM call
- **Performance metrics** - Task duration, depth, and success rates
- **Compilation artifacts** - Optimized prompts and few-shot examples

## MLflow Integration

### Configuration

Enable MLflow tracking in your configuration:

```yaml
# config/defaults/config.yaml
observability:
  mlflow:
    enabled: true
    tracking_uri: "http://127.0.0.1:5000"  # Local MLflow server
    experiment_name: "ROMA-DSPy"
    log_traces: true
    log_compiles: true
    log_evals: true
    log_traces_from_compile: false  # Expensive, disabled by default
```

Or via environment variables:

```bash
export MLFLOW_ENABLED=true
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT=ROMA-DSPy
```

### Starting MLflow Server

```bash
# Start MLflow UI
mlflow ui --port 5000

# Or with specific backend store
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Access the UI at http://localhost:5000

## What Gets Logged

### 1. Run-Level Metrics

Each solver execution creates an MLflow run with:

**Parameters:**
- `task` - The original goal/task
- `max_depth` - Maximum decomposition depth
- `config_version` - Configuration version
- `solver_type` - RecursiveSolver identifier

**Metrics:**
- `total_tasks` - Number of tasks created
- `completed_tasks` - Successfully completed tasks
- `failed_tasks` - Failed tasks
- `total_cost` - Total LLM API cost (USD)
- `total_tokens` - Total tokens consumed
- `execution_duration` - Total execution time (seconds)
- `max_depth_reached` - Actual maximum depth reached

### 2. LLM Traces

For each Language Model call:

**Logged Information:**
- Module name (atomizer, planner, executor, etc.)
- Model identifier (gpt-4, claude-3, etc.)
- Token usage (prompt, completion, total)
- Cost breakdown
- Latency (milliseconds)
- Input/output (if enabled)

**Metrics per module:**
- `{module}_calls` - Number of calls
- `{module}_tokens` - Total tokens
- `{module}_cost` - Total cost
- `{module}_avg_latency` - Average latency

### 3. Compilation Artifacts

When using DSPy optimization:

- Compiled predictor signatures
- Few-shot examples
- Optimization metrics
- Prompt templates

## Usage Examples

### Basic Usage

```python
from roma_dspy.config.manager import ConfigManager
from roma_dspy.core.engine.solve import RecursiveSolver

# Load config with MLflow enabled
config = ConfigManager(profile="high_quality").get_config()
config.observability.mlflow.enabled = True

# Create solver
solver = RecursiveSolver(config=config)

# Solve task - automatically logged to MLflow
result = await solver.async_solve("Plan a weekend in Barcelona")
```

### Custom Experiment Names

```python
config.observability.mlflow.experiment_name = "Barcelona-Planning-v2"
solver = RecursiveSolver(config=config)
```

### Programmatic Access

```python
from roma_dspy.observability.mlflow_manager import MLflowManager

# Initialize
mlflow_mgr = MLflowManager(config.observability.mlflow)
await mlflow_mgr.initialize()

# Start run
run_id = await mlflow_mgr.start_run(
    run_name="custom-run",
    tags={"version": "1.0", "experiment_type": "production"}
)

# Log metrics
await mlflow_mgr.log_metric("custom_metric", 42.0)
await mlflow_mgr.log_param("custom_param", "value")

# End run
await mlflow_mgr.end_run(status="FINISHED")
```

## Querying MLflow Data

### Using MLflow UI

1. Navigate to http://localhost:5000
2. Select your experiment
3. Compare runs, view metrics, download artifacts

### Using MLflow API

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Search runs
runs = mlflow.search_runs(
    experiment_names=["ROMA-DSPy"],
    filter_string="metrics.total_cost < 1.0",
    order_by=["metrics.execution_duration ASC"]
)

# Get best run
best_run = runs.sort_values("metrics.total_cost").iloc[0]
print(f"Best run: {best_run.run_id}")
print(f"Cost: ${best_run['metrics.total_cost']:.4f}")
```

### Analyzing Costs

```python
# Get all runs
runs = mlflow.search_runs(experiment_names=["ROMA-DSPy"])

# Cost analysis
total_cost = runs["metrics.total_cost"].sum()
avg_cost = runs["metrics.total_cost"].mean()
cost_by_depth = runs.groupby("params.max_depth")["metrics.total_cost"].mean()

print(f"Total spent: ${total_cost:.2f}")
print(f"Average per run: ${avg_cost:.4f}")
print("\nCost by depth:")
print(cost_by_depth)
```

## Cost Tracking

### Token Costs

ROMA-DSPy tracks costs for common LLM providers:

- **OpenAI**: gpt-4, gpt-3.5-turbo, etc.
- **Anthropic**: claude-3-opus, claude-3-sonnet, etc.
- **Fireworks AI**: Various models
- **OpenRouter**: Pass-through pricing

Costs are calculated using:
```python
cost = (prompt_tokens * prompt_price_per_1k / 1000) +
       (completion_tokens * completion_price_per_1k / 1000)
```

### Cost Optimization

Monitor these metrics to optimize costs:

1. **Tokens per task** - Identify verbose modules
2. **Failed task cost** - Wasted spend on failures
3. **Model selection** - Compare costs across models
4. **Depth vs cost** - Find optimal decomposition depth

## Performance Monitoring

### Key Metrics

**Latency:**
- Total execution time
- Per-module latency
- LLM call latency

**Throughput:**
- Tasks per minute
- Subtasks per decomposition
- Success rate

**Resource Usage:**
- Token consumption rate
- API calls per task
- Checkpoint frequency

### Alerts and Thresholds

Set up alerts for:

```python
# High cost runs
if run.metrics.total_cost > 5.0:
    alert("High cost run detected")

# Slow execution
if run.metrics.execution_duration > 300:
    alert("Slow execution")

# High failure rate
failure_rate = run.metrics.failed_tasks / run.metrics.total_tasks
if failure_rate > 0.2:
    alert("High failure rate")
```

## Integration with Postgres

When both MLflow and Postgres are enabled, you get dual observability:

**MLflow**: Experiment tracking, visualization, comparison
**Postgres**: Detailed execution traces, queryable history, audit trail

```python
# Query both sources
import mlflow
from roma_dspy.core.storage.postgres_storage import PostgresStorage

# MLflow - high-level metrics
runs = mlflow.search_runs(experiment_names=["ROMA-DSPy"])

# Postgres - detailed traces
storage = PostgresStorage(config.storage.postgres)
await storage.initialize()

for _, run in runs.iterrows():
    execution_id = run["tags.execution_id"]
    costs = await storage.get_execution_costs(execution_id)
    print(f"Run {execution_id}: ${costs['total_cost']:.4f}")
```

## Best Practices

1. **Use descriptive experiment names** - Organize by project/feature
2. **Tag runs appropriately** - Add version, environment, user tags
3. **Monitor costs regularly** - Set up cost alerts
4. **Archive old experiments** - Keep MLflow database manageable
5. **Disable expensive logging in production** - `log_traces_from_compile: false`
6. **Use remote tracking server** - For team collaboration
7. **Back up MLflow data** - Especially artifact stores

## Troubleshooting

### MLflow Connection Issues

```bash
# Check server is running
curl http://localhost:5000/health

# Check environment variable
echo $MLFLOW_TRACKING_URI

# Test connection
python -c "import mlflow; print(mlflow.get_tracking_uri())"
```

### Missing Metrics

```python
# Verify logging is enabled
print(config.observability.mlflow.enabled)
print(config.observability.mlflow.log_traces)

# Check MLflow manager initialization
print(solver.mlflow_manager._initialized)
```

### High Storage Usage

```bash
# Check artifact store size
du -sh ~/.mlflow

# Clean up old runs (use with caution)
mlflow gc --backend-store-uri sqlite:///mlflow.db
```

## Advanced Topics

### Custom Metrics

```python
# Add custom metrics to MLflow
async with solver.mlflow_manager.run_context():
    await solver.mlflow_manager.log_metric("custom_score", score)
    await solver.mlflow_manager.log_param("algorithm", "custom")
```

### Distributed Tracking

For multi-machine setups:

```yaml
observability:
  mlflow:
    tracking_uri: "http://mlflow-server.company.com:5000"
    # Use S3/GCS for artifacts
    artifact_location: "s3://my-bucket/mlflow-artifacts"
```

### Integration with Other Tools

MLflow integrates with:
- **Prometheus** - For operational metrics
- **Grafana** - For dashboards
- **Databricks** - For managed MLflow
- **Weights & Biases** - Via exporters

## Toolkit Metrics and Traceability

ROMA-DSPy provides comprehensive tracking of toolkit lifecycle and tool invocation metrics, enabling deep visibility into tool usage patterns, performance, and reliability.

### Overview

The toolkit metrics system automatically tracks:
- **Toolkit Lifecycle** - Creation, caching, cleanup operations
- **Tool Invocations** - Individual tool calls with timing and I/O metrics
- **Performance** - Duration, success rates, error patterns
- **Attribution** - Cost and usage per toolkit/tool

### Configuration

Enable toolkit metrics tracking:

```yaml
# config/defaults/config.yaml
observability:
  toolkit_metrics:
    enabled: true                    # Enable/disable tracking
    track_lifecycle: true            # Track toolkit operations
    track_invocations: true          # Track tool calls
    sample_rate: 1.0                 # Sample rate (0.0-1.0)
    persist_to_db: true              # Save to PostgreSQL
    persist_to_mlflow: false         # Save to MLflow
    batch_size: 100                  # Batch size for persistence
    async_persist: true              # Async persistence
```

Or via environment variables:

```bash
export TOOLKIT_METRICS_ENABLED=true
export TOOLKIT_TRACK_LIFECYCLE=true
export TOOLKIT_TRACK_INVOCATIONS=true
export TOOLKIT_SAMPLE_RATE=1.0
export TOOLKIT_PERSIST_DB=true
```

### What Gets Tracked

#### 1. Toolkit Lifecycle Events

**Tracked Operations:**
- `create` - Toolkit instantiation
- `cache_hit` - Retrieved from cache
- `cache_miss` - Cache lookup failed
- `cleanup` - Toolkit disposal

**Captured Data:**
- Operation timestamp
- Toolkit class name
- Duration (milliseconds)
- Success/failure status
- Error details (if failed)
- Custom metadata

#### 2. Tool Invocation Events

**Tracked for Each Call:**
- Tool name and toolkit class
- Invocation timestamp
- Duration (milliseconds)
- Input size (bytes)
- Output size (bytes)
- Success/failure status
- Error details (if failed)
- Custom metadata

### API Endpoints

Query toolkit metrics via REST API:

#### Get Aggregated Summary

```bash
curl http://localhost:8000/executions/{execution_id}/toolkit-metrics
```

**Response:**
```json
{
  "execution_id": "exec_123",
  "toolkit_lifecycle": {
    "total_created": 5,
    "cache_hit_rate": 0.75
  },
  "tool_invocations": {
    "total_calls": 50,
    "successful_calls": 48,
    "failed_calls": 2,
    "success_rate": 0.96,
    "avg_duration_ms": 125.5,
    "total_duration_ms": 6275.0
  },
  "by_toolkit": {
    "SerperToolkit": {
      "calls": 20,
      "successful": 20,
      "failed": 0,
      "avg_duration_ms": 150.0
    }
  },
  "by_tool": {
    "SerperToolkit.search_web": {
      "calls": 15,
      "successful": 15,
      "avg_duration_ms": 145.0
    }
  }
}
```

#### Get Raw Lifecycle Traces

```bash
# All lifecycle traces
curl http://localhost:8000/executions/{execution_id}/toolkit-traces

# Filter by operation
curl http://localhost:8000/executions/{execution_id}/toolkit-traces?operation=create

# Filter by toolkit class
curl http://localhost:8000/executions/{execution_id}/toolkit-traces?toolkit_class=SerperToolkit

# Limit results
curl http://localhost:8000/executions/{execution_id}/toolkit-traces?limit=100
```

#### Get Raw Tool Invocations

```bash
# All tool invocations
curl http://localhost:8000/executions/{execution_id}/tool-invocations

# Filter by toolkit
curl http://localhost:8000/executions/{execution_id}/tool-invocations?toolkit_class=SerperToolkit

# Filter by tool name
curl http://localhost:8000/executions/{execution_id}/tool-invocations?tool_name=search_web

# Combined filters
curl http://localhost:8000/executions/{execution_id}/tool-invocations?toolkit_class=SerperToolkit&tool_name=search_web
```

### Database Schema

#### toolkit_traces Table

```sql
CREATE TABLE toolkit_traces (
    trace_id BIGSERIAL PRIMARY KEY,
    execution_id VARCHAR(64) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    operation VARCHAR(32) NOT NULL,
    toolkit_class VARCHAR(128),
    duration_ms FLOAT NOT NULL,
    success BOOLEAN NOT NULL,
    error TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE
);

-- Indexes for query performance
CREATE INDEX idx_toolkit_traces_execution ON toolkit_traces (execution_id, timestamp);
CREATE INDEX idx_toolkit_traces_operation ON toolkit_traces (operation);
CREATE INDEX idx_toolkit_traces_toolkit_class ON toolkit_traces (toolkit_class);
CREATE INDEX idx_toolkit_traces_success ON toolkit_traces (success);
```

#### tool_invocation_traces Table

```sql
CREATE TABLE tool_invocation_traces (
    trace_id BIGSERIAL PRIMARY KEY,
    execution_id VARCHAR(64) NOT NULL,
    toolkit_class VARCHAR(128) NOT NULL,
    tool_name VARCHAR(128) NOT NULL,
    invoked_at TIMESTAMP WITH TIME ZONE NOT NULL,
    duration_ms FLOAT NOT NULL,
    input_size_bytes INTEGER NOT NULL,
    output_size_bytes INTEGER NOT NULL,
    success BOOLEAN NOT NULL,
    error TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE
);

-- Indexes for query performance
CREATE INDEX idx_tool_invocations_execution ON tool_invocation_traces (execution_id, invoked_at);
CREATE INDEX idx_tool_invocations_toolkit ON tool_invocation_traces (toolkit_class);
CREATE INDEX idx_tool_invocations_tool ON tool_invocation_traces (tool_name);
CREATE INDEX idx_tool_invocations_toolkit_tool ON tool_invocation_traces (toolkit_class, tool_name);
CREATE INDEX idx_tool_invocations_success ON tool_invocation_traces (success);
```

### Database Migration

Apply the migration to create toolkit metrics tables:

```bash
# Navigate to project root
cd /path/to/ROMA-DSPy

# Run migration
alembic upgrade head
```

Or manually:

```bash
# Check current version
alembic current

# Upgrade to toolkit metrics migration
alembic upgrade 004_toolkit_metrics

# Rollback if needed
alembic downgrade 003_add_dag_snapshot
```

### Usage Examples

#### Analyzing Toolkit Performance

```python
from roma_dspy.core.storage.postgres_storage import PostgresStorage

# Get toolkit metrics summary
summary = await storage.get_toolkit_metrics_summary("exec_123")

print(f"Total tool calls: {summary['tool_invocations']['total_calls']}")
print(f"Success rate: {summary['tool_invocations']['success_rate']:.2%}")
print(f"Average duration: {summary['tool_invocations']['avg_duration_ms']:.2f}ms")

# Analyze by toolkit
for toolkit, metrics in summary['by_toolkit'].items():
    print(f"\n{toolkit}:")
    print(f"  Calls: {metrics['calls']}")
    print(f"  Success rate: {metrics['successful'] / metrics['calls']:.2%}")
    print(f"  Avg duration: {metrics['avg_duration_ms']:.2f}ms")
```

#### Identifying Slow Tools

```python
# Get all tool invocations
invocations = await storage.get_tool_invocation_traces("exec_123")

# Sort by duration
slow_tools = sorted(invocations, key=lambda x: x.duration_ms, reverse=True)[:10]

print("Top 10 slowest tool calls:")
for inv in slow_tools:
    print(f"{inv.toolkit_class}.{inv.tool_name}: {inv.duration_ms:.2f}ms")
```

#### Tracking Failures

```python
# Get failed tool invocations
failed = await storage.get_tool_invocation_traces(
    execution_id="exec_123",
    limit=1000
)
failed = [inv for inv in failed if not inv.success]

# Group by error type
from collections import Counter
error_types = Counter(inv.metadata.get('error_type', 'Unknown') for inv in failed)

print("Failure breakdown:")
for error_type, count in error_types.most_common():
    print(f"  {error_type}: {count}")
```

#### Cache Performance Analysis

```python
# Get lifecycle traces
traces = await storage.get_toolkit_traces("exec_123")

# Calculate cache metrics
cache_hits = sum(1 for t in traces if t.operation == "cache_hit")
cache_misses = sum(1 for t in traces if t.operation == "cache_miss")
total = cache_hits + cache_misses

if total > 0:
    hit_rate = cache_hits / total
    print(f"Cache hit rate: {hit_rate:.2%}")
    print(f"Cache hits: {cache_hits}")
    print(f"Cache misses: {cache_misses}")
```

### Monitoring and Alerting

#### Key Metrics to Monitor

1. **Success Rate** - Alert if below 95%
2. **Average Duration** - Alert on significant increases
3. **Error Rate** - Alert on spikes
4. **Cache Hit Rate** - Alert if drops significantly

#### Example Prometheus Queries

```promql
# Success rate by toolkit
sum(rate(tool_invocations_success_total[5m])) by (toolkit_class)
/
sum(rate(tool_invocations_total[5m])) by (toolkit_class)

# P95 latency
histogram_quantile(0.95, sum(rate(tool_duration_ms_bucket[5m])) by (le, tool_name))

# Error rate
sum(rate(tool_invocations_failed_total[5m])) by (toolkit_class, error_type)
```

### Performance Tuning

#### Reduce Storage Overhead

```yaml
# Sample only 10% of calls in high-volume environments
observability:
  toolkit_metrics:
    sample_rate: 0.1
```

#### Batch Persistence

```yaml
# Increase batch size for better write performance
observability:
  toolkit_metrics:
    batch_size: 500
    async_persist: true
```

#### Disable Specific Tracking

```yaml
# Track only lifecycle, skip invocations
observability:
  toolkit_metrics:
    track_lifecycle: true
    track_invocations: false
```

### Troubleshooting

#### Metrics Not Appearing

1. **Check PostgreSQL is enabled:**
   ```yaml
   storage:
     postgres:
       enabled: true
   ```

2. **Verify migration applied:**
   ```bash
   alembic current
   # Should show: 004_toolkit_metrics (head)
   ```

3. **Check configuration:**
   ```python
   print(config.observability.toolkit_metrics.enabled)
   print(config.observability.toolkit_metrics.persist_to_db)
   ```

#### High Storage Usage

```sql
-- Check table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE tablename LIKE '%trace%'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Clean old executions (careful!)
DELETE FROM executions WHERE created_at < NOW() - INTERVAL '30 days';
```

#### Missing Traces

```python
# Check if context is properly set
from roma_dspy.core.context import ExecutionContext

ctx = ExecutionContext.get()
if ctx:
    print(f"Execution ID: {ctx.execution_id}")
    print(f"Toolkit events: {len(ctx.toolkit_events)}")
    print(f"Tool invocations: {len(ctx.tool_invocations)}")
else:
    print("No ExecutionContext found!")
```

### Best Practices

1. **Enable in Development** - Use full tracking (sample_rate=1.0)
2. **Sample in Production** - Use lower sample rates for high-volume systems
3. **Monitor Key Metrics** - Set up alerts on success rates and latencies
4. **Regular Cleanup** - Archive or delete old execution data
5. **Index Management** - Monitor index size and query performance
6. **Correlate with LM Traces** - Combine with LM metrics for cost attribution

### Integration with MLflow

```python
# Log toolkit metrics to MLflow
from roma_dspy.core.observability import MLflowManager

async with mlflow_manager.run_context():
    summary = await storage.get_toolkit_metrics_summary(execution_id)

    # Log aggregate metrics
    await mlflow_manager.log_metric(
        "toolkit_success_rate",
        summary["tool_invocations"]["success_rate"]
    )
    await mlflow_manager.log_metric(
        "avg_tool_duration_ms",
        summary["tool_invocations"]["avg_duration_ms"]
    )

    # Log per-toolkit metrics
    for toolkit, metrics in summary["by_toolkit"].items():
        await mlflow_manager.log_metric(
            f"{toolkit}_calls",
            metrics["calls"]
        )
```

## Further Reading

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking Guide](https://mlflow.org/docs/latest/tracking.html)
- [DSPy Observability](https://dspy-docs.vercel.app/)
- [PostgreSQL Performance Tuning](https://www.postgresql.org/docs/current/performance-tips.html)
