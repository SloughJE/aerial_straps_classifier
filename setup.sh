#!/bin/bash
if [ "$ENVIRONMENT" = "development" ]; then
    echo "Development environment, skipping script setup"
else
    ./start.sh
fi