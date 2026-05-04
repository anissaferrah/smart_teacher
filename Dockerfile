FROM python:3.11-slim

WORKDIR /app

# Dépendances système pour audio + unstructured
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-ara \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Patch layer — packages utilisés par le code mais oubliés dans requirements.txt
# (à fusionner dans requirements.txt lors du prochain pin de dépendances)
RUN pip install --no-cache-dir asyncpg langgraph

COPY . .

RUN mkdir -p courses logs data/rag_cache media/audio media/pdfs media/slides static

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]