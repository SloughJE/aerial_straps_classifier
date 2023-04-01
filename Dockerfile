FROM python:3.8-slim-buster

RUN mkdir /code
WORKDIR /code

ENV PYTHONUNBUFFERED 1

RUN apt-get -y update && \
    apt-get -y install bc curl gnupg2 jq less unzip wget libglib2.0-0 git gcc python3-dev libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /code/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
