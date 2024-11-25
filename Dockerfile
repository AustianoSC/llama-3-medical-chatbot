# Use the NVIDIA CUDA base image with the desired version of CUDA and cuDNN
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt update -y \
    && apt install -y python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY ./src/ ./src/
COPY ./main.py .

# Set the entry point for your application
ENTRYPOINT ["python3", "main.py"]