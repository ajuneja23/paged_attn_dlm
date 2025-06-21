FROM debian:bullseye

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


WORKDIR /workspace

COPY . .


CMD ["/bin/bash"]
