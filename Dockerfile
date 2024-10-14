# Start with Debian Bookworm Slim as the base image for building FFmpeg
FROM debian:bookworm-slim AS ffmpeg

# Install necessary build dependencies
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends \
    build-essential \
    git \
    pkg-config \
    yasm \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone FFmpeg repository
RUN git clone https://github.com/FFmpeg/FFmpeg.git --depth 1 --branch n6.1.1 --single-branch /FFmpeg-6.1.1

# Set working directory to FFmpeg source
WORKDIR /FFmpeg-6.1.1

# Configure and build FFmpeg
RUN PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
      --prefix="$HOME/ffmpeg_build" \
      --pkg-config-flags="--static" \
      --extra-cflags="-I$HOME/ffmpeg_build/include" \
      --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
      --extra-libs="-lpthread -lm" \
      --ld="g++" \
      --bindir="$HOME/bin" \
      --disable-doc \
      --disable-htmlpages \
      --disable-podpages \
      --disable-txtpages \
      --disable-network \
      --disable-autodetect \
      --disable-hwaccels \
      --disable-ffprobe \
      --disable-ffplay \
      --enable-filter=copy \
      --enable-protocol=file \
      --enable-small && \
    PATH="$HOME/bin:$PATH" make -j$(nproc) && \
    make install && \
    hash -r

# Use NVIDIA CUDA image as the final base
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHON_VERSION=3.10
ENV POETRY_VENV=/app/.venv

# Install Python and pip
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set up Python symlinks
RUN ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -s -f /usr/bin/pip3 /usr/bin/pip

# Set up Poetry virtual environment
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==1.6.1

# Add Poetry to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"

# Set working directory
WORKDIR /app

# Copy Poetry files
COPY poetry.lock pyproject.toml ./

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.in-project true
RUN poetry install --no-root

# Copy application files
COPY . .

# Copy FFmpeg
COPY --from=ffmpeg /FFmpeg-6.1.1 /FFmpeg-6.1.1
COPY --from=ffmpeg /root/bin/ffmpeg /usr/local/bin/ffmpeg

# Install application and PyTorch
RUN poetry install
RUN $POETRY_VENV/bin/pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch

# Expose port for the application
EXPOSE 9000

# Set the command to run the application
CMD whisper-asr-webservice
