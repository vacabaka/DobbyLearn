# ROMA-DSPy Test Suite

This directory contains the test suite for ROMA-DSPy, organized by test type and categorized with pytest markers for flexible test execution.

## Test Organization

```
tests/
├── unit/              # Fast, isolated unit tests
├── integration/       # Integration tests with external services
├── tools/             # Toolkit-specific tests
├── validation/        # Validation and verification tests
├── performance/       # Performance benchmarks (future)
└── fixtures/          # Shared test fixtures
```

## Test Markers

Tests are categorized using pytest markers. Use markers to run specific subsets of tests:

### Primary Categories
- `unit` - Fast unit tests with no external dependencies
- `integration` - Integration tests requiring external services
- `e2e` - End-to-end system tests

### Requirement Markers
- `requires_db` - Requires PostgreSQL database
- `requires_llm` - Requires LLM API keys (OpenAI, etc.)
- `requires_e2b` - Requires E2B sandbox environment

### Feature Markers
- `checkpoint` - Checkpoint/recovery functionality tests
- `error_handling` - Error propagation tests
- `tools` - Toolkit integration tests
- `performance` - Performance benchmarks
- `slow` - Long-running tests

## Running Tests

### Run all tests
```bash
pytest
```

### Run only unit tests (fast)
```bash
pytest -m unit
```

### Run integration tests (requires services)
```bash
pytest -m integration
```

### Run tests that require PostgreSQL
```bash
# Start Postgres first
docker-compose up -d postgres

# Run database tests
pytest -m requires_db

# Cleanup
docker-compose down
```

### Run tests that require LLM APIs
```bash
# Set API keys
export OPENAI_API_KEY=your_key_here

# Run LLM tests
pytest -m requires_llm
```

### Run specific test categories
```bash
# Only checkpoint tests
pytest -m checkpoint

# Only toolkit tests
pytest -m tools

# Integration tests that don't need DB
pytest -m "integration and not requires_db"

# E2E tests with all requirements
pytest -m "e2e and requires_db and requires_llm"
```

### Run tests by directory
```bash
# All unit tests
pytest tests/unit/

# Specific test file
pytest tests/unit/test_dag_serialization.py

# Specific test function
pytest tests/unit/test_dag_serialization.py::test_serialize_task_node
```

### Test Coverage
```bash
# Run with coverage report
pytest --cov=src/roma_dspy --cov-report=html

# Open coverage report
open htmlcov/index.html
```

## Setting Up Test Environment

### 1. Install Development Dependencies
```bash
pip install -e ".[dev]"
```

### 2. Start PostgreSQL (for DB tests)
```bash
docker-compose up -d postgres

# Verify it's running
docker-compose ps

# Check logs
docker-compose logs postgres
```

### 3. Set Environment Variables
```bash
# Required for LLM tests
export OPENAI_API_KEY=sk-...
export FIREWORKS_API_KEY=...

# Required for DB tests (docker-compose defaults)
export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost/roma_dspy_test

# Optional: E2B sandbox
export E2B_API_KEY=...
```

### 4. Run Database Migrations (first time)
```bash
# Apply migrations to test database
uv run alembic upgrade head
```

## Writing Tests

### Test Structure
```python
import pytest

@pytest.mark.unit
def test_my_unit_test():
    """Test description."""
    # Fast test with no external dependencies
    assert True

@pytest.mark.integration
@pytest.mark.requires_db
async def test_my_integration_test(postgres_storage):
    """Test description."""
    # Integration test using fixtures
    result = await postgres_storage.get_execution("exec_123")
    assert result is not None
```

### Using Markers
```python
# Single marker
@pytest.mark.unit

# Multiple markers
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_db

# Skip with condition
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY environment variable"
)
```

### Fixtures
Common fixtures are available in `tests/conftest.py` and `tests/fixtures/`:

- `postgres_storage` - Initialized PostgresStorage instance
- `postgres_config` - PostgresConfig for testing
- `temp_checkpoint_dir` - Temporary directory for checkpoint tests
- Mock fixtures for LMs and external services

## Continuous Integration

Tests run automatically on:
- Pull requests (unit + integration without external deps)
- Main branch commits (full suite with services)

See `.github/workflows/ci.yml` for CI configuration.

## Troubleshooting

### Tests Timing Out
```bash
# Increase timeout for slow tests
pytest --timeout=300
```

### Database Connection Errors
```bash
# Check Postgres is running
docker-compose ps

# Reset database
docker-compose down -v
docker-compose up -d postgres
```

### Import Errors
```bash
# Reinstall in editable mode
pip install -e .
```

### Skipped Tests
```bash
# See why tests were skipped
pytest -v -rs

# Force run skipped tests (dangerous!)
pytest --runxfail
```

## Test Best Practices

1. **Keep unit tests fast** - No I/O, no network, no external services
2. **Use appropriate markers** - Tag tests accurately for selective running
3. **Mock external dependencies** - Use mocks for LLMs in unit tests
4. **Clean up resources** - Use fixtures for setup/teardown
5. **Test edge cases** - Invalid inputs, error conditions, boundary values
6. **Document test purpose** - Clear docstrings explaining what's being tested

## Performance Testing

Performance tests are planned for future development:

```bash
# Run performance benchmarks (future)
pytest -m performance --benchmark-only
```

## Test Data

Test data and fixtures are in:
- `tests/fixtures/` - Reusable test data
- Individual test files - Test-specific data

Avoid committing sensitive data (API keys, credentials) to test files.
