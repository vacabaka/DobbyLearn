# ROMA-DSPy E2B Sandbox Template with S3 Integration
# Base image: e2bdev/code-interpreter (Debian-based with Python)
FROM e2bdev/code-interpreter:latest

# Build arguments for configuration
ARG AWS_REGION=us-east-1
ARG S3_MOUNT_DIR=/opt/sentient
ARG S3_BUCKET_NAME
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

# Set environment variables (including credentials - required for goofys at runtime)
# Note: Credentials must be in ENV for goofys to work, not just in ~/.aws/credentials
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_REGION=${AWS_REGION}
ENV STORAGE_BASE_PATH=${S3_MOUNT_DIR}
ENV S3_BUCKET_NAME=${S3_BUCKET_NAME}

# Update package list
RUN apt-get update

# Install essential tools for S3 filesystem mounting
RUN apt-get install -y \
    curl \
    wget \
    fuse \
    ca-certificates \
    jq \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Install goofys for high-performance S3 mounting
# Using the latest release for optimal performance
RUN curl -L https://github.com/kahing/goofys/releases/latest/download/goofys -o /usr/local/bin/goofys \
    && chmod +x /usr/local/bin/goofys

# Set up FUSE permissions for non-root users
RUN echo "user_allow_other" >> /etc/fuse.conf

# Setup AWS credentials from build args (visible in one layer, but needed for E2B CLI compatibility)
# NOTE: Credentials will be visible in docker history for this layer, but E2B CLI doesn't support --secret
RUN mkdir -p /root/.aws /home/user/.aws && \
    if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then \
        echo "[default]" > /root/.aws/credentials && \
        echo "aws_access_key_id = ${AWS_ACCESS_KEY_ID}" >> /root/.aws/credentials && \
        echo "aws_secret_access_key = ${AWS_SECRET_ACCESS_KEY}" >> /root/.aws/credentials && \
        echo "[default]" > /root/.aws/config && \
        echo "region = ${AWS_REGION}" >> /root/.aws/config && \
        echo "output = json" >> /root/.aws/config && \
        cp /root/.aws/credentials /home/user/.aws/credentials && \
        cp /root/.aws/config /home/user/.aws/config && \
        chmod 600 /root/.aws/credentials /home/user/.aws/credentials && \
        echo "AWS credentials configured from build args"; \
    else \
        echo "No AWS credentials provided - S3 mounting will be skipped"; \
    fi

# Copy requirements file for additional Python dependencies
COPY requirements.txt /tmp/requirements.txt

# Install additional Python packages not in base image
# Using --no-cache-dir to minimize image size
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Create essential directories
RUN mkdir -p /workspace ${S3_MOUNT_DIR}

# Copy startup script to E2B's expected location
# E2B runs scripts from /root/.jupyter/ on sandbox initialization
COPY start-up.sh /root/.jupyter/start-up.sh

# Make startup script executable
RUN chmod +x /root/.jupyter/start-up.sh

# Set workspace as working directory
WORKDIR /workspace

# Note: E2B automatically executes /root/.jupyter/start-up.sh via start_cmd in e2b.toml
# The script will:
# 1. Mount S3 bucket to STORAGE_BASE_PATH using goofys
# 2. Create executions directory structure
# 3. Verify write access
# 4. Set up Python path