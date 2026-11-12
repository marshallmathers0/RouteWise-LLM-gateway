# RouteWise on Hugging Face Spaces (Docker SDK): API + Streamlit in one container.
#   - uvicorn serves the gateway on 127.0.0.1:8000 (internal only)
#   - Streamlit (playground + dashboard) is the public app on port 7860
# No Redis (exact cache off, in-memory rate limiting / breaker); FAISS + SQLite semantic
# cache; request logs in SQLite. Spaces' disk is ephemeral, so both start from demo seeds.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    HOME=/app
WORKDIR /app

RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt requirements-ui.txt ./
RUN pip install --no-cache-dir -r requirements-ui.txt

# Bake the models into the image so a cold start doesn't download them.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/quora-distilroberta-base')"

COPY routewise ./routewise
COPY config ./config
COPY dashboard ./dashboard
COPY scripts ./scripts
COPY artifacts ./artifacts
COPY data/demo ./data/demo
COPY deploy/hf_space/start.sh ./start.sh

# Spaces runs the container as uid 1000; make the writable dirs its own.
RUN chmod +x start.sh && mkdir -p /app/runtime && chown -R 1000:1000 /app
USER 1000

EXPOSE 7860
CMD ["./start.sh"]
