FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /data/models && chown -R 1000:1000 /data/models

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    sox \
    libsox-fmt-all \
    portaudio19-dev \
    pandoc \
    software-properties-common \
    curl \
    git \
    poppler-utils \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.11 python3.11-distutils python3.11-dev python3.11-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default python and python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --set python /usr/bin/python3.11 \
    && update-alternatives --set python3 /usr/bin/python3.11

# Install pip for Python 3.11
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.11 get-pip.py \
    && rm get-pip.py

# Verify Python and pip versions
RUN python --version && python3 --version && pip --version && pip3 --version

WORKDIR /app

# Install Python dependencies individually
RUN pip3 install cython
RUN pip3 install numpy
RUN pip3 install typing_extensions
RUN pip3 install torch
RUN pip3 install torchaudio
RUN pip3 install pyannote.audio
RUN pip3 install transformers
RUN pip3 install fastapi
RUN pip3 install uvicorn
RUN pip3 install python-multipart

# Copy application files
COPY . .

# Create a non-root user and switch to it
RUN useradd -m appuser
USER appuser

# Expose port
EXPOSE 9000

# Run the FastAPI app
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]

