#!/bin/bash
# Start the API server from the correct directory

cd "$(dirname "$0")" || exit 1

# Set API key if not already set
if [ -z "$OPENAI_API_KEY" ]; then
    # Load from main .env file or use environment variable
    if [ -f "../.env" ]; then
        export $(grep -v '^#' ../.env | xargs)
    elif [ -z "$OPENAI_API_KEY" ]; then
        echo "⚠️  Warning: OPENAI_API_KEY not set. Set it in ../.env or export it."
    fi
fi

# Kill any existing server on port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null

# Start the server
echo "Starting server from: $(pwd)"
echo "API Key: ${OPENAI_API_KEY:0:20}..."
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

