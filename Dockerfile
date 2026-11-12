FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache
WORKDIR /app

# CPU-only torch first: the default PyPI wheel bundles CUDA (~2.5 GB extra) we don't need.
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt requirements-ui.txt ./
RUN pip install --no-cache-dir -r requirements-ui.txt

# Bake the embedding model + verifier into the image so a cold start doesn't download them.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/quora-distilroberta-base')"

COPY routewise ./routewise
COPY config ./config
COPY dashboard ./dashboard
COPY scripts ./scripts
COPY artifacts ./artifacts
COPY data/demo ./data/demo

EXPOSE 8000 8501
CMD ["uvicorn", "routewise.main:app", "--host", "0.0.0.0", "--port", "8000"]
