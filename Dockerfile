# Use Debian with just the build tools
FROM debian:bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch (no CUDA needed)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Set working directory
WORKDIR /workspace

# Copy your source files
COPY . .


# Default command
CMD ["/bin/bash"]
