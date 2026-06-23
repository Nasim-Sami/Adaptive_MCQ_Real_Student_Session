#!/bin/bash
set -e

ollama serve &

for i in $(seq 1 30); do
  if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    break
  fi
  sleep 1
done

exec python app.py
