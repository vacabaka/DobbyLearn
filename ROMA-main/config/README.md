# ROMA-DSPy Configuration System

This directory contains the YAML-based configuration system for ROMA-DSPy, built with OmegaConf for configuration operations and Pydantic for validation.

## Quick Start

```python
from roma_dspy.config import load_config

# Load with defaults
config = load_config()

# Load with profile
config = load_config(profile="lightweight")

# Load with overrides
config = load_config(overrides=["agents.executor.llm.temperature=0.5"])
```

## Configuration Structure

### Base Configuration (`defaults/config.yaml`)
Default settings that override Pydantic defaults:
- Project metadata
- Agent configuration overrides
- Runtime settings
- Resilience parameters

### Profiles (`profiles/*.yaml`)
Delta configurations that overlay specific use cases:
- **lightweight.yaml**: Reduced resource usage, lower token limits
- **tool_enabled.yaml**: Prepared for future tool implementation

## Configuration Schema

### LLM Configuration
```yaml
agents:
  executor:
    llm:
      model: "gpt-4o-mini"
      temperature: 0.7
      max_tokens: 2000
      timeout: 30
      api_key: ${oc.env:OPENAI_API_KEY}  # Environment variable
```

### Agent Configuration
```yaml
agents:
  executor:
    prediction_strategy: "chain_of_thought"
    tools: []
    enabled: true
    agent_config:          # Agent business logic parameters
      max_subtasks: 10
    strategy_config: {}    # Prediction strategy algorithm parameters
```

### Runtime Configuration
```yaml
runtime:
  max_concurrency: 5
  timeout: 30
  verbose: ${oc.env:ROMA_VERBOSE,false}
  cache_dir: ".cache/dspy"
```

### Resilience Configuration
```yaml
resilience:
  max_retries: 3
  retry_delay: 1.0
  circuit_breaker_threshold: 5
  circuit_breaker_timeout: 60
```

## Configuration Resolution Order

Later sources override earlier ones:
1. **Pydantic defaults** (in schema classes)
2. **Base YAML** (`defaults/config.yaml`)
3. **Profile YAML** (`profiles/{profile}.yaml`)
4. **Override strings** (`["key=value"]`)
5. **Environment variables** (`ROMA_*`)

## Environment Variables

### Naming Convention
- Prefix: `ROMA_`
- Nested keys: double underscore `__`
- Example: `ROMA_AGENTS__EXECUTOR__LLM__TEMPERATURE=0.5`

### Common Variables
```bash
# API Keys
export OPENAI_API_KEY="your-key"
export FIREWORKS_API_KEY="your-key"

# Runtime settings
export ROMA_VERBOSE=true
export ROMA_MAX_RETRIES=5
export ROMA_CACHE_DIR="/custom/cache"

# Agent settings
export ROMA_AGENTS__EXECUTOR__LLM__TEMPERATURE=0.3
```

## Profile Usage

### Creating Custom Profiles
Create `profiles/my_profile.yaml`:
```yaml
# My custom profile
agents:
  executor:
    llm:
      temperature: 0.1
    agent_config:
      max_iterations: 20

runtime:
  max_concurrency: 10
```

### Using Profiles
```python
config = load_config(profile="my_profile")
```

## Advanced Features

### OmegaConf Interpolation
```yaml
# Variable interpolation
base_timeout: 30
runtime:
  timeout: ${base_timeout}

# Environment variable with default
cache_dir: ${oc.env:ROMA_CACHE_DIR,.cache/dspy}
```

### Configuration Caching
The ConfigManager automatically caches loaded configurations for performance:
```python
manager = ConfigManager()
config1 = manager.load_config()  # Loads from file
config2 = manager.load_config()  # Uses cache
manager.clear_cache()            # Clears cache
```

### Validation

The system provides two-stage validation:
1. **OmegaConf**: YAML structure and type checking
2. **Pydantic**: Business logic validation

Example validations:
- Temperature must be between 0.0 and 2.0
- max_tokens must be between 1 and 100,000
- Tool-strategy compatibility checking
- Timeout consistency validation

## Module Integration

### BaseModule Integration
```python
from roma_dspy.config import load_config
from roma_dspy.core.modules import Executor

# Load configuration
config = load_config(profile="lightweight")

# Create module with config
executor = Executor(
    signature=MySignature,
    config=config.agents.executor
)
```

### RecursiveSolver Integration
```python
from roma_dspy.core.engine.solve import RecursiveSolver

# Create solver with config
solver = RecursiveSolver(config=config)
result = solver.solve("Complex task")
```

## Configuration Files

- `defaults/config.yaml` - Base configuration overrides
- `profiles/lightweight.yaml` - Minimal resource usage
- `profiles/tool_enabled.yaml` - Tool-ready configuration

## Best Practices

1. **Use profiles** for different deployment environments
2. **Environment variables** for secrets and environment-specific settings
3. **Override strings** for quick testing and experimentation
4. **Base config** for organization-wide defaults
5. **Separate agent_config and strategy_config** for proper parameter isolation

## Troubleshooting

### Common Issues
- **OmegaConf type errors**: Check YAML syntax and avoid Pydantic Field objects
- **Validation errors**: Review Pydantic validators and constraints
- **Missing profiles**: Ensure profile files exist in `profiles/` directory
- **Environment variables**: Use correct naming convention with `ROMA_` prefix

### Debugging
```python
# Enable verbose logging
config = load_config(overrides=["runtime.verbose=true"])

# Check resolved configuration
print(OmegaConf.to_yaml(config))

# Validate specific sections
from roma_dspy.config.schemas import LLMConfig
llm_config = LLMConfig(**config.agents.executor.llm)
```