# ROMA-DSPy Configuration Examples

This directory contains curated examples demonstrating key concepts and patterns for configuring ROMA-DSPy agents.

## Directory Structure

```
config/examples/
├── basic/              # Basic concepts and patterns
├── mcp/                # Model Context Protocol (MCP) servers
├── crypto/             # Crypto/finance domain examples
├── advanced/           # Advanced patterns and techniques
└── prompts/            # Custom prompt templates (Jinja)
```

## Quick Start

### 1. Minimal Configuration
```bash
uv run python -m roma_dspy.cli solve "Your task here" --config config/examples/basic/minimal.yaml
```

### 2. Try MCP Public Server (No Setup!)
```bash
uv run python -m roma_dspy.cli solve "What is the current price of Bitcoin?" --config config/examples/mcp/http_public_server.yaml
```

## Examples by Category

### Basic (`basic/`)
**Concepts**: Fundamentals, toolkit usage, multi-toolkit patterns

| Example | Demonstrates | Setup Required |
|---------|-------------|----------------|
| `minimal.yaml` | Simplest possible configuration | ❌ No |
| `multi_toolkit.yaml` | Combining multiple toolkits | E2B API key |

**Key Learnings:**
- How to configure agents and toolkits
- Combining multiple tools in one agent
- Basic runtime settings

### MCP (`mcp/`)
**Concepts**: MCP servers, HTTP vs stdio, multi-server orchestration

| Example | Demonstrates | Setup Required |
|---------|-------------|----------------|
| `http_public_server.yaml` | Public HTTP MCP server | ❌ No |
| `stdio_local_server.yaml` | Local stdio MCP server | npm install |
| `multi_server.yaml` | Multiple MCP servers | npm + API keys |

**Key Learnings:**
- MCP HTTP servers (remote, no installation)
- MCP stdio servers (local subprocess)
- Combining multiple MCP servers
- Storage configuration for large data

### Crypto (`crypto/`)
**Concepts**: Real-world domain-specific agents

| Example | Demonstrates | Setup Required |
|---------|-------------|----------------|
| `crypto_agent.yaml` | Comprehensive crypto analysis | Optional API keys |

**Key Learnings:**
- Combining MCP + native toolkits
- Multi-source data aggregation
- Domain-specific agent design

### Advanced (`advanced/`)
**Concepts**: Advanced patterns, optimization, customization

| Example | Demonstrates | Setup Required |
|---------|-------------|----------------|
| `task_aware_mapping.yaml` | Task-specific executor configs | API keys |
| `custom_prompts.yaml` | Custom prompts and demos | ❌ No |

**Key Learnings:**
- Task-aware agent mapping (RETRIEVE, CODE_INTERPRET, THINK, WRITE)
- Cost/quality optimization per task type
- Loading custom signature instructions
- Few-shot learning with demos

## Configuration Patterns

### Basic Agent Structure
```yaml
agents:
  executor:
    llm:
      model: openai/gpt-4o-mini
      temperature: 0.3
      max_tokens: 2000
    prediction_strategy: react  # Required for tools
    toolkits:
      - class_name: ToolkitName
        enabled: true
        toolkit_config:
          # Toolkit-specific settings
```

### Task-Aware Mapping
```yaml
agents:
  executor:
    # Default configuration

agent_mapping:
  executors:
    RETRIEVE:
      # Fast model + web search
    CODE_INTERPRET:
      # Powerful model + code execution
    THINK:
      # Reasoning-focused
    WRITE:
      # Creative writing
```

### Custom Prompts
```yaml
agents:
  executor:
    signature_instructions: "module.path:PROMPT_VAR"
    demos: "module.path:DEMOS_VAR"
```

## Environment Variables

Required for specific examples:

```bash
# E2B (code execution)
export E2B_API_KEY=your_key

# Exa (web search via MCP)
export EXA_API_KEY=your_key

# GitHub MCP server
export GITHUB_PERSONAL_ACCESS_TOKEN=your_token

# Serper (web search toolkit)
export SERPER_API_KEY=your_key

# OpenRouter (recommended LLM provider)
export OPENROUTER_API_KEY=your_key
```

## Available Toolkits

### Native Toolkits (Built into ROMA-DSPy)

| Toolkit | Purpose | API Key Required |
|---------|---------|------------------|
| **FileToolkit** | File operations | ❌ No |
| **CalculatorToolkit** | Math operations | ❌ No |
| **E2BToolkit** | Code execution | ✅ Yes |
| **SerperToolkit** | Web search | ✅ Yes |
| **BinanceToolkit** | Crypto market data | ❌ No (public endpoints) |
| **CoinGeckoToolkit** | Crypto prices | ❌ No (public) |
| **DefiLlamaToolkit** | DeFi protocol data | ❌ No |
| **ArkhamToolkit** | Blockchain analytics | ❌ No |

### MCP Toolkits (via MCPToolkit)

**Public HTTP Servers** (no setup):
- CoinGecko: `https://mcp.api.coingecko.com/sse`
- Exa: `https://mcp.exa.ai/mcp` (requires API key)

**NPM Stdio Servers** (require `npm install -g`):
- Filesystem: `@modelcontextprotocol/server-filesystem`
- GitHub: `@modelcontextprotocol/server-github`
- SQLite: `@modelcontextprotocol/server-sqlite`
- Slack: `@modelcontextprotocol/server-slack`

## Tips for Success

### 1. Start Simple
Begin with `basic/minimal.yaml`, then add complexity.

### 2. Use Public MCP Servers First
Try `mcp/http_public_server.yaml` - no installation needed!

### 3. Task-Aware Mapping for Cost Optimization
Use different models for different task types:
- RETRIEVE: Fast, cheap models
- CODE_INTERPRET: Powerful models
- THINK: Reasoning-focused
- WRITE: Creative models

### 4. Enable Storage for Large Data
```yaml
use_storage: true
storage_threshold_kb: 100  # Store results > 100KB
```

### 5. Custom Prompts for Better Performance
Load optimized prompts from `prompt_optimization/seed_prompts/`

## Common Issues

### "Unknown toolkit class"
- Check spelling of `class_name`
- Ensure toolkit is imported/registered

### "Tools don't support strategy"
- Use `prediction_strategy: react` for tool usage
- `chain_of_thought` doesn't support tools

### "API key required"
- Set environment variable: `export API_KEY=value`
- Or use `${oc.env:API_KEY}` in config

### MCP Server Connection Failed
- **HTTP**: Check URL and network
- **Stdio**: Ensure npm package installed globally

## Next Steps

1. **Copy and modify** examples for your use case
2. **Combine patterns** from different examples
3. **See profiles** in `config/profiles/` for complete configurations
4. **Read seed prompts** in `prompt_optimization/seed_prompts/` for inspiration

## Resources

- **Main Documentation**: `/CLAUDE.md`
- **Agent Profiles**: `config/profiles/`
- **Seed Prompts**: `prompt_optimization/seed_prompts/`
- **MCP Documentation**: https://modelcontextprotocol.io/
- **Awesome MCP Servers**: https://github.com/wong2/awesome-mcp-servers