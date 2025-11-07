# E2B Sandbox Template with Secure S3 Mounting

This directory contains the E2B sandbox template configuration for ROMA-DSPy with secure AWS credential handling for S3 storage mounting.

## Overview

E2B (Execution Environment 2B) provides isolated sandbox environments for code execution. This template builds a custom sandbox with:
- Pre-installed goofys for S3 filesystem mounting
- Secure AWS credential handling via Docker build secrets
- S3 storage mounted during template build and available in all sandbox instances

## Security Architecture

### Docker Build Secrets (Secure)
This implementation uses Docker BuildKit secrets to pass AWS credentials during template build:

1. **Credentials are passed as temporary files** during `docker build`
2. **Credentials are NOT stored in image layers** (can be verified with `docker history`)
3. **Credentials are used once during build** to create `~/.aws/credentials` in the template
4. **S3 is mounted during build** and the template is snapshotted with S3 already mounted

### Why Not Environment Variables?
Using `ARG` or `ENV` for credentials would bake them into Docker image layers, making them:
- Visible in `docker history`
- Extractable from the image
- A security risk if the image is shared

## Files

### `e2b.toml`
E2B configuration file specifying:
- Template ID and name
- `start_cmd`: Command to run during template build (points to `start-up.sh`)
- Dockerfile location

### `e2b.Dockerfile`
Multi-stage Dockerfile that:
1. Installs system dependencies (curl, wget, fuse, goofys)
2. Uses `RUN --mount=type=secret` to read AWS credentials from Docker secrets
3. Creates `~/.aws/credentials` without storing credentials in image layers
4. Copies and executes the startup script via `start_cmd`

### `start-up.sh`
Startup script that executes during template build to:
1. Check if AWS credentials exist (from Dockerfile Docker secrets)
2. Mount S3 bucket using goofys with the credentials
3. Verify mount and create directory structure
4. Test write access

### `requirements.txt`
Python dependencies to install in the sandbox (currently empty, can be extended)

## E2B Execution Model

**CRITICAL UNDERSTANDING:**

```
Template Build (once)              Runtime (each instance)
─────────────────────              ───────────────────────
1. Docker build                    1. Restore from snapshot
2. Install packages                2. Use pre-mounted S3
3. Run start_cmd                   3. Execute user code
4. Execute start-up.sh
5. Mount S3 with goofys
6. Create snapshot
```

- **start_cmd runs during template BUILD**, not at runtime
- The sandbox is **snapshotted** after the startup script completes
- Runtime instances **restore from the snapshot** with S3 already mounted
- The startup script does **NOT re-run** on each sandbox instance

## Setup Process

### 1. Prerequisites
```bash
# Install E2B CLI
npm install -g @e2b/cli

# Login to E2B
e2b login

# Ensure Docker BuildKit is enabled
export DOCKER_BUILDKIT=1
```

### 2. Environment Variables
Ensure these are set in `.env`:
```bash
E2B_API_KEY=your_e2b_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
ROMA_S3_BUCKET=your-bucket-name
```

### 3. Build Template

The `setup.sh` script handles template building automatically:

```bash
cd /Users/barannama/ROMA-DSPy
./setup.sh
```

Or manually:
```bash
cd docker/e2b

# Create temporary secret files
mkdir -p /tmp/e2b-secrets
echo -n "$AWS_ACCESS_KEY_ID" > /tmp/e2b-secrets/aws_access_key_id
echo -n "$AWS_SECRET_ACCESS_KEY" > /tmp/e2b-secrets/aws_secret_access_key

# Build with Docker secrets
DOCKER_BUILDKIT=1 e2b template build \
  --build-arg AWS_REGION="us-east-1" \
  --build-arg S3_BUCKET_NAME="your-bucket-name" \
  --secret id=aws_access_key_id,src=/tmp/e2b-secrets/aws_access_key_id \
  --secret id=aws_secret_access_key,src=/tmp/e2b-secrets/aws_secret_access_key

# Clean up
rm -rf /tmp/e2b-secrets
```

## Verification

ROMA-DSPy includes a comprehensive integration test suite for E2B template validation.

### Run Validation Tests

```bash
# Via justfile (recommended)
just e2b-validate

# Or directly with pytest
pytest tests/integration/test_e2b_template_validation.py -v
```

### What the Tests Validate

The integration test suite (`tests/integration/test_e2b_template_validation.py`) checks:

1. **Prerequisites**
   - ✅ E2B CLI installed
   - ✅ E2B authentication configured

2. **Template Status**
   - ✅ Template exists in E2B
   - ✅ AWS credentials NOT visible in Docker image layers (security check)

3. **Sandbox Functionality**
   - ✅ Sandbox creation succeeds
   - ✅ S3 storage mounted at `/opt/sentient`
   - ✅ Write access to S3 works
   - ✅ Read-back from S3 works
   - ✅ Cleanup works

### Quick Manual Checks

```bash
# List templates
e2b template list

# Get template info
e2b template info roma-dspy-sandbox

# Quick sandbox test
just e2b-test
```

### Manual Testing (if needed)

```python
from e2b import Sandbox

# Create sandbox instance
sandbox = Sandbox("roma-dspy-sandbox")

# Test S3 access
result = sandbox.run_code("import os; print(os.listdir('/opt/sentient'))")
print(result.stdout)

# Test write access
result = sandbox.run_code("""
import os
test_file = '/opt/sentient/executions/test.txt'
with open(test_file, 'w') as f:
    f.write('test')
print(f'Written to {test_file}')
""")
print(result.stdout)

sandbox.close()
```

## Build Logs

During template build, you should see:
```
[E2B-SETUP] 🚀 Starting ROMA-DSPy E2B Sandbox setup...
[E2B-SETUP] Storage path: /opt/sentient
[E2B-SETUP] S3 bucket: your-bucket-name
[E2B-SETUP] ✅ AWS credentials files found (configured via Docker secrets)
[E2B-SETUP] Mounting S3 bucket: your-bucket-name to /opt/sentient
[E2B-SETUP] Attempting goofys mount...
[E2B-SETUP] ✅ S3 bucket mounted successfully with goofys
[E2B-SETUP] ✅ S3 mount verified
[E2B-SETUP] ✅ Write access confirmed
[E2B-SETUP] ✅ Startup script completed successfully
```

## Troubleshooting

### Build Fails: "AWS credentials not found"
- Ensure `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are set in `.env`
- Check that `setup.sh` is loading `.env` correctly
- Verify secret files are being created in the temporary directory

### Build Fails: "goofys mount failed"
- Check AWS credentials have S3 access permissions
- Verify S3 bucket exists and is accessible from the region
- Check bucket name is correct in `ROMA_S3_BUCKET`

### Build Succeeds but S3 Not Accessible
- Template might have built without S3 mounting (check logs)
- Rebuild template with correct credentials
- Use `e2b template info` to check template status

### Credentials Visible in Image
- Ensure `DOCKER_BUILDKIT=1` is set before building
- Check that `--secret` flags are being used (not `--build-arg` for credentials)
- Verify using `docker history` that credentials don't appear

## Differences from Sentient Implementation

| Aspect | Sentient (Old) | ROMA-DSPy (New) |
|--------|---------------|-----------------|
| Credential Passing | `ARG` and `ENV` | Docker secrets |
| Security | ❌ Baked into image | ✅ Not stored in image |
| Build Command | Simple `e2b template build` | BuildKit with `--secret` flags |
| Runtime Mounting | Env vars at runtime | Pre-mounted at build time |
| Credential Visibility | Visible in `docker history` | Not visible anywhere |

## Integration with ROMA-DSPy

The E2B toolkit (`src/roma_dspy/tools/core/e2b.py`) uses this template:

```python
from roma_dspy.tools.core.e2b import E2BToolkit

# E2BToolkit automatically uses the roma-dspy-sandbox template
toolkit = E2BToolkit(template="roma-dspy-sandbox")

# Execute code with S3 access
result = toolkit.execute_python("""
import os
# S3 is already mounted at /opt/sentient
files = os.listdir('/opt/sentient/executions')
print(f'Found {len(files)} execution directories')
""")
```

## References

- [E2B Documentation](https://e2b.dev/docs)
- [Docker Build Secrets](https://docs.docker.com/build/building/secrets/)
- [goofys GitHub](https://github.com/kahing/goofys)
- [E2B Templates Guide](https://e2b.dev/docs/guide/templates)