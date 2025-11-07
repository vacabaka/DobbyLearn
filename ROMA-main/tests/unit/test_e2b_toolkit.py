"""Tests for E2B toolkit."""

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.roma_dspy.tools.core.e2b import E2BToolkit


class MockExecution:
    """Mock execution result from E2B sandbox."""

    def __init__(self, results=None, stdout=None, stderr=None, error=None):
        self.results = results or [Mock(text="42")]
        self.logs = Mock()
        self.logs.stdout = stdout or []
        self.logs.stderr = stderr or []
        self.error = error


class MockCommandResult:
    """Mock command result for E2B commands.run()."""

    def __init__(self, exit_code=0, stdout="", stderr="", error=None):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.error = error


class MockSandbox:
    """Mock E2B Sandbox for testing."""

    def __init__(self, sandbox_id="test-sandbox-123"):
        self.sandbox_id = sandbox_id
        self._running = True
        self.files = Mock()
        self.commands = Mock()

    def is_running(self):
        """Mock is_running check."""
        return self._running

    def run_code(self, code):
        """Mock code execution."""
        return MockExecution()

    def kill(self):
        """Mock sandbox kill."""
        self._running = False

    def get_host(self, port):
        """Mock get_host."""
        return f"https://test-sandbox.e2b.dev:{port}"


@pytest.fixture
def mock_e2b():
    """Mock E2B module for all tests."""
    # Mock the import at the builtins level
    mock_sandbox_class = Mock()
    mock_e2b_module = Mock()
    mock_e2b_module.Sandbox = mock_sandbox_class

    with patch.dict('sys.modules', {'e2b_code_interpreter': mock_e2b_module}):
        yield mock_sandbox_class


class TestE2BToolkit:
    """Test E2BToolkit functionality."""

    @patch.dict(os.environ, {'E2B_API_KEY': 'test_api_key_12345'})
    def test_initialization_with_api_key(self, mock_e2b):
        """Test toolkit initialization with API key."""
        toolkit = E2BToolkit(timeout=600)

        assert toolkit.api_key == 'test_api_key_12345'
        assert toolkit.timeout == 600
        assert toolkit.template == 'roma-dspy-sandbox'  # Default template when no E2B_TEMPLATE_ID is set
        assert toolkit.auto_reinitialize is True

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key(self, mock_e2b):
        """Test toolkit fails without API key."""
        with pytest.raises(ValueError, match="E2B_API_KEY is required"):
            E2BToolkit()

    @patch.dict(os.environ, {'E2B_API_KEY': 'test_key'})
    def test_missing_dependency(self):
        """Test toolkit fails without e2b library."""
        # Mock the e2b_code_interpreter import to fail
        with patch.dict('sys.modules', {'e2b_code_interpreter': None}):
            with pytest.raises(ImportError, match="e2b-code-interpreter library is required"):
                E2BToolkit()

    @patch.dict(os.environ, {'E2B_API_KEY': 'test_api_key_12345'})
    def test_create_sandbox(self, mock_e2b):
        """Test sandbox creation."""
        mock_sandbox = MockSandbox()
        mock_e2b.create.return_value = mock_sandbox

        toolkit = E2BToolkit()

        with toolkit._lock:
            sandbox = toolkit._create_sandbox()

        assert sandbox == mock_sandbox
        assert toolkit._sandbox_id == "test-sandbox-123"
        assert toolkit._created_at > 0

    @patch.dict(os.environ, {'E2B_API_KEY': 'test_api_key_12345'})
    def test_ensure_sandbox_alive_creates_if_none(self, mock_e2b):
        """Test _ensure_sandbox_alive creates sandbox if none exists."""
        mock_sandbox = MockSandbox()
        mock_e2b.create.return_value = mock_sandbox

        toolkit = E2BToolkit()
        sandbox = toolkit._ensure_sandbox_alive()

        assert sandbox is not None
        assert toolkit._sandbox_id == "test-sandbox-123"

    @patch.dict(os.environ, {'E2B_API_KEY': 'test_api_key_12345'})
    def test_ensure_sandbox_alive_returns_running_sandbox(self, mock_e2b):
        """Test _ensure_sandbox_alive returns existing running sandbox."""
        mock_sandbox = MockSandbox()
        mock_e2b.create.return_value = mock_sandbox

        toolkit = E2BToolkit()
        toolkit._ensure_sandbox_alive()
        create_calls = mock_e2b.create.call_count

        sandbox = toolkit._ensure_sandbox_alive()

        assert sandbox == mock_sandbox
        assert mock_e2b.create.call_count == create_calls

    @patch.dict(os.environ, {'E2B_API_KEY': 'test_api_key_12345'})
    def test_ensure_sandbox_alive_reinitializes_dead_sandbox(self, mock_e2b):
        """Test _ensure_sandbox_alive reinitializes dead sandbox."""
        mock_sandbox = MockSandbox()
        mock_e2b.create.return_value = mock_sandbox

        toolkit = E2BToolkit()
        toolkit._ensure_sandbox_alive()

        mock_sandbox._running = False
        new_mock_sandbox = MockSandbox("new-sandbox-456")
        mock_e2b.create.return_value = new_mock_sandbox

        sandbox = toolkit._ensure_sandbox_alive()

        assert sandbox == new_mock_sandbox
        assert toolkit._sandbox_id == "new-sandbox-456"

    @patch.dict(os.environ, {'E2B_API_KEY': 'test_api_key_12345'})
    def test_run_python_code(self, mock_e2b):
        """Test Python code execution."""
        mock_sandbox = MockSandbox()
        mock_e2b.create.return_value = mock_sandbox

        toolkit = E2BToolkit()
        result = toolkit.run_python_code("print('hello')")
        data = json.loads(result)

        assert data["success"] is True
        assert "42" in data["results"]
        assert data["sandbox_id"] == "test-sandbox-123"

    @patch.dict(os.environ, {'E2B_API_KEY': 'test_api_key_12345'})
    def test_run_python_code_error_handling(self, mock_e2b):
        """Test error handling in code execution."""
        mock_sandbox = Mock()
        mock_sandbox.id = "test-sandbox-123"
        mock_sandbox.is_running.return_value = True
        mock_sandbox.run_code.side_effect = Exception("Execution failed")
        mock_e2b.create.return_value = mock_sandbox

        toolkit = E2BToolkit()
        result = toolkit.run_python_code("bad code")
        data = json.loads(result)

        assert data["success"] is False
        assert "Execution failed" in data["error"]

    @patch.dict(os.environ, {'E2B_API_KEY': 'test_api_key_12345'})
    def test_run_command(self, mock_e2b):
        """Test command execution."""
        mock_sandbox = MockSandbox()
        mock_result = MockCommandResult(exit_code=0, stdout="Success!", stderr="")
        mock_sandbox.commands.run.return_value = mock_result
        mock_e2b.create.return_value = mock_sandbox

        toolkit = E2BToolkit()
        result = toolkit.run_command("echo hello")
        data = json.loads(result)

        assert data["success"] is True
        assert data["exit_code"] == 0
        assert data["stdout"] == "Success!"

    @patch.dict(os.environ, {'E2B_API_KEY': 'test_api_key_12345'})
    def test_get_sandbox_status_no_sandbox(self, mock_e2b):
        """Test status when no sandbox exists."""
        toolkit = E2BToolkit()
        result = toolkit.get_sandbox_status()
        data = json.loads(result)

        assert data["success"] is True
        assert data["status"] == "no_sandbox"

    @patch.dict(os.environ, {'E2B_API_KEY': 'test_api_key_12345'})
    def test_get_sandbox_status_running(self, mock_e2b):
        """Test status of running sandbox."""
        mock_sandbox = MockSandbox()
        mock_e2b.create.return_value = mock_sandbox

        toolkit = E2BToolkit()
        toolkit._ensure_sandbox_alive()

        result = toolkit.get_sandbox_status()
        data = json.loads(result)

        assert data["success"] is True
        assert data["status"] == "running"
        assert data["sandbox_id"] == "test-sandbox-123"

    @patch.dict(os.environ, {'E2B_API_KEY': 'test_api_key_12345'})
    def test_restart_sandbox(self, mock_e2b):
        """Test manual sandbox restart."""
        mock_sandbox = MockSandbox()
        mock_e2b.create.return_value = mock_sandbox

        toolkit = E2BToolkit()
        toolkit._ensure_sandbox_alive()
        old_id = toolkit._sandbox_id

        new_sandbox = MockSandbox("restarted-sandbox")
        mock_e2b.create.return_value = new_sandbox

        result = toolkit.restart_sandbox()
        data = json.loads(result)

        assert data["success"] is True
        assert data["old_sandbox_id"] == old_id

    @patch.dict(os.environ, {'E2B_API_KEY': 'test_api_key_12345'})
    def test_upload_file(self, mock_e2b):
        """Test file upload to sandbox."""
        mock_sandbox = MockSandbox()
        mock_e2b.create.return_value = mock_sandbox

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            toolkit = E2BToolkit()
            result = toolkit.upload_file(temp_path, "/home/user/test.txt")
            data = json.loads(result)

            assert data["success"] is True
            assert data["remote_path"] == "/home/user/test.txt"
        finally:
            Path(temp_path).unlink()

    @patch.dict(os.environ, {'E2B_API_KEY': 'test_api_key_12345'})
    def test_download_file(self, mock_e2b):
        """Test file download from sandbox."""
        mock_sandbox = MockSandbox()
        mock_sandbox.files.read.return_value = b"downloaded content"
        mock_e2b.create.return_value = mock_sandbox

        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / "downloaded.txt"
            toolkit = E2BToolkit()
            result = toolkit.download_file("/home/user/file.txt", str(local_path))
            data = json.loads(result)

            assert data["success"] is True
            assert local_path.exists()

    @patch.dict(os.environ, {'E2B_API_KEY': 'test_api_key_12345'})
    def test_thread_safety(self, mock_e2b):
        """Test thread-safe concurrent operations."""
        mock_sandbox = MockSandbox()
        mock_e2b.create.return_value = mock_sandbox

        toolkit = E2BToolkit()
        results = []

        def run_code():
            result = toolkit.run_python_code("x = 1")
            results.append(result)

        threads = [threading.Thread(target=run_code) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        for result in results:
            data = json.loads(result)
            assert data["success"] is True

    @patch.dict(os.environ, {'E2B_API_KEY': 'test_api_key_12345'})
    def test_cleanup_on_destruction(self, mock_e2b):
        """Test sandbox cleanup when toolkit is destroyed."""
        mock_sandbox = MockSandbox()
        mock_e2b.create.return_value = mock_sandbox

        toolkit = E2BToolkit()
        toolkit._ensure_sandbox_alive()

        toolkit.__del__()

        assert mock_sandbox._running is False

    @patch.dict(os.environ, {'E2B_API_KEY': 'test_api_key_12345'})
    def test_auto_reinitialize_disabled(self, mock_e2b):
        """Test behavior when auto_reinitialize is disabled."""
        mock_sandbox = MockSandbox()
        mock_e2b.create.return_value = mock_sandbox

        toolkit = E2BToolkit(auto_reinitialize=False)
        toolkit._ensure_sandbox_alive()

        mock_sandbox._running = False

        with pytest.raises(RuntimeError, match="Sandbox died"):
            toolkit._ensure_sandbox_alive()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])