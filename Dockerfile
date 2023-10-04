# Use a specific version for reproducibility
FROM python:3.8-slim-buster

# Build-time argument for the environment setting
ARG ENVIRONMENT=production
ENV APP_ENVIRONMENT=$ENVIRONMENT

# Environment settings
ENV PYTHONUNBUFFERED=1

# Create a working directory
WORKDIR /code

# Installing dependencies
# Chain commands to reduce layers and clean up in the same step
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 gcc python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy just the requirements file and install dependencies
# Use --no-cache-dir to avoid caching and reduce image size
COPY requirements.txt requirements_dev.txt /code/
RUN if [ "$ENVIRONMENT" = "development" ]; then \
        pip install --no-cache-dir -r requirements_dev.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy all code into the image
COPY . /code/

# Make the script executable
RUN chmod +x /code/setup.sh /code/start.sh

# Specify the default thing to run when starting a container from this image
CMD ["/code/setup.sh"]