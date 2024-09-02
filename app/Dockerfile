FROM python:3.11.8
# Set the working directory in the container
WORKDIR /api

# Install necessary system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0

# Copy the current directory contents into the container at /app
COPY . /api


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port available to the world outside this container
EXPOSE 8080

# Run your FastAPI app
CMD ["uvicorn", "main:app", "--reload", "--port=8080", "--host=0.0.0.0"]