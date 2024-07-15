# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies and clean up
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create necessary directories and copy model/data files
RUN mkdir -p saved_model storage
COPY saved_model/caption_model.pth saved_model/
COPY storage/word_to_idx.pkl storage/
COPY storage/idx_to_word.pkl storage/

# Expose the application port
EXPOSE 8080

# Define environment variable
ENV FLASK_APP=app.py

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
