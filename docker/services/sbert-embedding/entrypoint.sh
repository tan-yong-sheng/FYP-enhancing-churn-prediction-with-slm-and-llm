#!/bin/sh
set -e  # Exit if any command fails

# Run the Uvicorn command with environment variables
exec uvicorn main:app --host "$HOST" --port "$PORT" --workers "$WORKERS"