# Use the slim Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update; apt-get install -y build-essential; rm -rf /var/lib/apt/lists/*

# Install pip and upgrade it
RUN pip install --upgrade pip

# Install necessary Python dependencies including CPU-only PyTorch
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install numpy
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install catboost

# Copy your Python scripts and data to the container
COPY step1.py /app/
COPY step2.py /app/
COPY utils.py /app/
COPY run.py /app/
COPY candidates_data.csv /app/
COPY spacecraft_images /app/spacecraft_images

# Command to run your application
CMD ["python3", "run.py"]
