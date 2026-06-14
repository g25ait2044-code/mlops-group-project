# Start with a lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Accept the Hugging Face model name as a build argument
ARG HF_MODEL_NAME=your-hf-username/my-mlops-model
ENV HF_MODEL_NAME=$HF_MODEL_NAME

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your inference script
COPY src/inference.py .

# Run inference when the container starts
CMD ["python", "inference.py"]