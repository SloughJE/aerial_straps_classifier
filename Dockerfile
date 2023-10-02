# Use a specific version for reproducibility
FROM python:3.8.12-slim-buster

# Build-time argument for the environment setting
ARG ENVIRONMENT=production
ENV APP_ENVIRONMENT $ENVIRONMENT
# Create a working directory
RUN mkdir /code 
WORKDIR /code

# Environment settings
ENV PYTHONUNBUFFERED 1

# Installing dependencies
# Chain commands to reduce layers and clean up in the same step
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 gcc python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy just the requirements file and install dependencies
# Use --no-cache-dir to avoid caching and reduce image size
COPY requirements.txt requirements_dev.txt /code/
# Conditionally install development or production requirements
RUN if [ "$ENVIRONMENT" = "development" ]; then \
        pip install --no-cache-dir -r requirements_dev.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Conditionally copy code and set up start script
COPY . /source
RUN chmod +x /source/setup.sh
CMD ["/source/setup.sh"]