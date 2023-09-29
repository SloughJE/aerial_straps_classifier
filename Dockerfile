# Use a specific version for reproducibility
FROM python:3.8.12-slim-buster

# Build-time argument for the environment setting
#ARG ENVIRONMENT=production

# Create a working directory
RUN mkdir /code 
WORKDIR /code

# Environment settings
ENV PYTHONUNBUFFERED 1
ENV APP_ENVIRONMENT $ENVIRONMENT

# Installing dependencies
# Chain commands to reduce layers and clean up in the same step
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 gcc python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy just the requirements file and install dependencies
# Use --no-cache-dir to avoid caching and reduce image size
COPY requirements.txt /code/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code into the docker container
COPY . /code/

# Make the start script executable
RUN chmod +x /code/start.sh

# Set the command to run the start script
CMD ["sh", "/code/start.sh"]
