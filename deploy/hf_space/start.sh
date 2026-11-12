#!/bin/sh
# Start the RouteWise gateway (internal) and the Streamlit UI (public, port 7860).
set -e
RUNTIME=/app/runtime
mkdir -p "$RUNTIME"

# Writable copies of the demo seeds (Spaces' disk resets on every restart).
[ -f "$RUNTIME/semantic_cache.db" ] || cp /app/data/demo/semantic_cache.db "$RUNTIME/semantic_cache.db" 2>/dev/null || true
sed "s#semantic_db_path: data/semantic_cache.db#semantic_db_path: $RUNTIME/semantic_cache.db#" \
    /app/config/routewise.yaml > "$RUNTIME/routewise.yaml"

export ROUTEWISE_CONFIG="$RUNTIME/routewise.yaml"
export DATABASE_URL="sqlite:///$RUNTIME/routewise.db"
export REDIS_URL=""
export ROUTEWISE_API_KEYS="${ROUTEWISE_API_KEYS:-rw-demo-key}"
export ROUTEWISE_API_KEY="${ROUTEWISE_API_KEYS%%,*}"
export ROUTEWISE_URL="http://127.0.0.1:8000"
export DEMO_LOGS_DB=/app/data/demo/logs.db

uvicorn routewise.main:app --host 127.0.0.1 --port 8000 &

# Wait for the gateway (model loading takes ~10-20 s on a free CPU Space).
for i in $(seq 1 90); do
  python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" 2>/dev/null && break
  sleep 1
done

exec streamlit run dashboard/app.py --server.port 7860 --server.address 0.0.0.0 \
     --server.headless true --browser.gatherUsageStats false
