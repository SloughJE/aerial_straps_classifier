#!/bin/bash

if [ "$ENVIRONMENT" = "development" ]; then
    echo "Development environment, skipping copy and script setup"
else
    cp -r /source/* /code/
    chmod +x /code/start.sh
    /code/start.sh
fi
