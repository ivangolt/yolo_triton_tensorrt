# FROM python:3.11.8
# # Set the working directory in the container
# WORKDIR /app

# # Install necessary system dependencies for OpenCV
# RUN apt-get update && \
#     apt-get install -y libgl1-mesa-glx libglib2.0-0

# # Copy the current directory contents into the container at /app
# COPY . /app


# # Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Make port available to the world outside this container
# EXPOSE 8080

# # Run your FastAPI app
# CMD ["uvicorn", "main:app", "--reload", "--port=8080", "--host=0.0.0.0"]

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

ENV PYTHONPATH="$PYTHONPATH:/app" 
# Copy the FastAPI service code into the container
COPY . .

# Install any necessary dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn opencv-python-headless && \
    pip install --no-cache-dir tritonclient[all]

# Expose the port the app runs on
EXPOSE 8080

# Command to run the FastAPI application
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
