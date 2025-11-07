# E2B Integration Setup Guide

## Overview

This guide explains how to set up E2B code execution sandboxes with S3 storage integration for ROMA-DSPy. The setup enables agents to execute code in isolated sandboxes while maintaining access to shared S3 storage via goofys.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Host System                          │
│                                                              │
│  ┌────────────┐         ┌─────────────────────────────┐    │
│  │   Agent    │────────>│   Toolkit (e.g., DefiLlama) │    │
│  └────────────┘         └─────────────────────────────┘    │
│         │                         │                          │
│         │                         │                          │
│         │                         ▼                          │
│         │              ┌─────────────────────┐              │
│         │              │   FileStorage       │              │
│         │              │   (S3 via goofys)   │              │
│         │              └─────────────────────┘              │
│         │                         │                          │
│         ▼                         │                          │
│  ┌────────────┐                  │                          │
│  │ E2BToolkit │                  │                          │
│  └────────────┘                  │                          │
│         │                         │                          │
└─────────┼─────────────────────────┼──────────────────────────┘
          │                         │
          │     Same S3 Bucket      │
          │                         │
┌─────────▼─────────────────────────▼──────────────────────────┐
│                      E2B Sandbox                              │
│                                                               │
│  ┌───────────────────────────────────────────────────┐      │
│  │  start-up.sh: Mounts S3 to /opt/sentient         │      │
│  │  via goofys using env vars from host              │      │
│  └───────────────────────────────────────────────────┘      │
│                                                               │
│  ┌───────────────────────────────────────────────────┐      │
│  │  Agent-Generated Code Executes                     │      │
│  │  - Reads from /opt/sentient/executions/...        │      │
│  │  - Writes to /opt/sentient/executions/...         │      │
│  └───────────────────────────────────────────────────┘      │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **E2B Account**
   - Sign up at [e2b.dev](https://e2b.dev)
   - Get API key from dashboard

2. **AWS S3 Bucket**
   - Create S3 bucket for storage
   - Configure AWS credentials with S3 access

3. **E2B CLI** (for template creation)
   ```bash
   npm install -g @e2b/cli
   # or
   yarn global add @e2b/cli
   ```

## Step 1: Environment Configuration

### 1.1 Configure Environment Variables

Copy `.env.example` to `.env` and fill in:

```bash
# Storage Configuration
STORAGE_BASE_PATH=/opt/sentient
ROMA_S3_BUCKET=your-s3-bucket-name
AWS_REGION=us-east-1

# AWS Credentials
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key

# E2B Configuration
E2B_API_KEY=your_e2b_api_key
E2B_TEMPLATE_ID=roma-dspy-sandbox
```

### 1.2 Verify Configuration

```python
from roma_dspy.config.manager import ConfigManager

config = ConfigManager().load_config()
print(f"Storage path: {config.storage.base_path}")
print(f"S3 bucket: {os.getenv('ROMA_S3_BUCKET')}")
```

## Step 2: Create Custom E2B Template

E2B templates define the sandbox environment. We need a custom template that includes our S3 mounting script.

### 2.1 Initialize Template

```bash
cd /Users/barannama/ROMA-DSPy

# Initialize E2B template
e2b template init roma-dspy-sandbox
```

This creates `.e2b/` directory with template configuration.

### 2.2 Add Startup Script

Copy our startup script to the template:

```bash
# Copy start-up.sh to template
cp docker/e2b/start-up.sh .e2b/start-up.sh

# Verify script is executable
chmod +x .e2b/start-up.sh
```

The `start-up.sh` script:
- Installs goofys in the sandbox
- Mounts S3 bucket to `$STORAGE_BASE_PATH`
- Validates write access
- Sets up Python dependencies

### 2.3 Build and Publish Template

```bash
# Build template (creates Docker image)
e2b template build

# This will output a template ID like:
# ✓ Template built successfully
# Template ID: roma-dspy-sandbox-abc123

# Copy the template ID to your .env file
echo "E2B_TEMPLATE_ID=<your-template-id>" >> .env
```

### 2.4 Verify Template

```bash
# List your templates
e2b template list

# You should see your template with ID matching .env
```

## Step 3: Local Storage Setup

### 3.1 Run Local Setup Script

```bash
# Make script executable
chmod +x scripts/setup_local.sh

# Run setup (mounts S3 locally via goofys)
./scripts/setup_local.sh
```

This script:
1. Installs goofys (if not present)
2. Mounts S3 bucket to local path
3. Creates symlink if needed
4. Validates write access

### 3.2 Verify Local Storage

```bash
# Check mount
mount | grep goofys

# Verify directory structure
ls -la /opt/sentient/executions/

# Test write access
echo "test" > /opt/sentient/executions/test.txt
cat /opt/sentient/executions/test.txt
rm /opt/sentient/executions/test.txt
```

## Step 4: Test E2B Integration

### 4.1 Basic Test

```python
from roma_dspy.tools.core import E2BToolkit
from roma_dspy.config.manager import ConfigManager
from roma_dspy.core.storage import FileStorage

# Load config
config = ConfigManager().load_config()

# Create storage
storage = FileStorage(
    config=config.storage,
    execution_id="test_e2b_001"
)

# Write file on host
test_data = b"Hello from host!"
await storage.put("test.txt", test_data)
print(f"Wrote to: {storage.get_artifacts_path('test.txt')}")

# Create E2B toolkit
e2b = E2BToolkit()

# Read file in E2B sandbox
code = f"""
import os
file_path = '{storage.get_artifacts_path('test.txt')}'
print(f'Reading from: {{file_path}}')
with open(file_path, 'r') as f:
    content = f.read()
    print(f'Content: {{content}}')
"""

result = e2b.run_python_code(code)
print(result)
```

Expected output:
```json
{
  "success": true,
  "results": [],
  "stdout": [
    "Reading from: /opt/sentient/executions/test_e2b_001/artifacts/test.txt",
    "Content: Hello from host!"
  ],
  "stderr": [],
  "error": null,
  "sandbox_id": "..."
}
```

### 4.2 Run Integration Tests

```bash
# Run E2B integration tests
pytest tests/integration/test_e2b_integration.py -v

# Run E2E storage test
pytest tests/integration/test_e2e_storage.py -v
```

## Step 5: Production Deployment

### 5.1 Environment-Specific Configuration

**Development (.env.development)**:
```bash
STORAGE_BASE_PATH=/opt/sentient/dev
ROMA_S3_BUCKET=roma-storage-dev
E2B_TEMPLATE_ID=roma-dspy-sandbox-dev
```

**Production (.env.production)**:
```bash
STORAGE_BASE_PATH=/opt/sentient/prod
ROMA_S3_BUCKET=roma-storage-prod
E2B_TEMPLATE_ID=roma-dspy-sandbox-prod
```

### 5.2 Update Template

When updating startup script:

```bash
# Modify docker/e2b/start-up.sh
# Then update template:

cp docker/e2b/start-up.sh .e2b/start-up.sh
e2b template build
```

## Troubleshooting

### Issue: Sandbox Can't Access S3

**Symptoms**: E2B code execution fails with file not found errors

**Solution**:
1. Verify env vars are passed to sandbox:
   ```python
   e2b = E2BToolkit()
   status = e2b.get_sandbox_status()
   print(status)  # Check env vars
   ```

2. Check start-up.sh logs in E2B dashboard

3. Verify AWS credentials are valid:
   ```bash
   aws s3 ls s3://$ROMA_S3_BUCKET
   ```

### Issue: goofys Mount Fails

**Symptoms**: Local setup script fails or mount point empty

**Solution**:
1. Check AWS credentials:
   ```bash
   aws sts get-caller-identity
   ```

2. Verify S3 bucket exists:
   ```bash
   aws s3 ls | grep $ROMA_S3_BUCKET
   ```

3. Check goofys installation:
   ```bash
   which goofys
   goofys --version
   ```

### Issue: Path Mismatch Between Host and E2B

**Symptoms**: Files written on host not visible in E2B

**Solution**:
1. Verify `STORAGE_BASE_PATH` is same in:
   - `.env` file
   - Local setup script output
   - E2B start-up.sh

2. Check both systems point to same S3 bucket:
   ```bash
   # Local
   mount | grep goofys

   # E2B (run in sandbox)
   mount | grep goofys
   ```

### Issue: Template Not Found

**Symptoms**: E2B toolkit fails with template not found

**Solution**:
1. Verify template exists:
   ```bash
   e2b template list
   ```

2. Check `E2B_TEMPLATE_ID` in .env matches template ID

3. Rebuild template if needed:
   ```bash
   e2b template build
   ```

## Advanced Configuration

### Custom Goofys Options

Edit `start-up.sh` to customize goofys mount:

```bash
# In start-up.sh, modify goofys command:
goofys \
    --region "${AWS_REGION}" \
    --stat-cache-ttl 5m \           # Longer cache
    --type-cache-ttl 5m \
    --max-idle-handles 1000 \       # More file handles
    --dir-mode 0755 \
    --file-mode 0644 \
    "${S3_BUCKET}" \
    "${STORAGE_BASE_PATH}"
```

### Multiple E2B Templates

Create environment-specific templates:

```bash
# Development template
e2b template init roma-dspy-dev
cp docker/e2b/start-up.sh .e2b/start-up.sh
e2b template build

# Production template (with optimizations)
e2b template init roma-dspy-prod
# Edit .e2b/start-up.sh with production settings
e2b template build
```

### Monitoring Storage Usage

```python
from roma_dspy.core.storage import FileStorage

storage = FileStorage(config=config.storage, execution_id="exec_123")
info = await storage.get_storage_info()

print(f"Total size: {info['total_size_mb']} MB")
print(f"File count: {info['file_count']}")
```

## Best Practices

1. **Use Execution IDs**: Always scope storage to execution IDs for isolation

2. **Clean Up Temp Files**: Use `cleanup_temp_files()` after execution:
   ```python
   await storage.cleanup_execution_temp_files()
   ```

3. **Monitor Costs**: Track S3 storage and E2B sandbox usage

4. **Template Versioning**: Version E2B templates in production:
   ```bash
   e2b template build --name roma-dspy-prod-v1.0.0
   ```

5. **Error Handling**: Always check E2B execution results:
   ```python
   result = e2b.run_python_code(code)
   result_data = json.loads(result)
   if not result_data["success"]:
       logger.error(f"E2B execution failed: {result_data['error']}")
   ```

## References

- [E2B Documentation](https://e2b.dev/docs)
- [Goofys GitHub](https://github.com/kahing/goofys)
- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [ROMA-DSPy Storage Architecture](/docs/STORAGE_ARCHITECTURE.md)