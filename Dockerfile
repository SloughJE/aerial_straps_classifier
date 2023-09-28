FROM python:3.8-slim-buster

# Build-time argument for the environment setting
ARG ENVIRONMENT=production

RUN mkdir /code
WORKDIR /code

# Environment settings
ENV PYTHONUNBUFFERED 1
ENV APP_ENVIRONMENT $ENVIRONMENT

# Installing dependencies
RUN apt-get -y update && \
    apt-get -y install libgl1-mesa-glx libglib2.0-0 gcc python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Installing Python requirements
COPY requirements${APP_ENVIRONMENT:+_dev}.txt /code/
RUN pip install --upgrade pip && \
    pip install -r requirements${APP_ENVIRONMENT:+_dev}.txt

# Conditionally setting the FastAPI run command for production
CMD if [ "$APP_ENVIRONMENT" = "production" ]; then uvicorn api.main:app --host 0.0.0.0 --port 80; else /bin/bash; fi
