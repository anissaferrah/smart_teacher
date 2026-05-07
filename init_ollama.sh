#!/bin/bash
# Initialize Ollama with text + vision models on first startup.
# Both pulls are idempotent — Ollama no-ops if the model is already present.

echo "🚀 Initializing Ollama models..."

# Wait for Ollama to be ready
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is ready"
        break
    fi
    echo "⏳ Waiting for Ollama... ($i/30)"
    sleep 2
done

# 1. Text fallback (used by Brain when OpenAI is unavailable).
#    Model name comes from $OLLAMA_TEXT_MODEL (matches Config.OLLAMA_TEXT_MODEL).
TEXT_MODEL="${OLLAMA_TEXT_MODEL:-qwen2.5}"
echo "📥 Pulling text model: $TEXT_MODEL ..."
ollama pull "$TEXT_MODEL"

# 2. Vision fallback (used by services/vision_describe.py for slide
#    concept extraction when OpenAI gpt-4o-mini is unavailable). The
#    .env variable OLLAMA_VISION_MODEL points to this model — keep them
#    in sync. Default "moondream" is compact (~1.6 GB disk, ~2-3 GB
#    RAM) and fits on a 15 GB-RAM machine alongside Smart Teacher's
#    other services. For a higher-quality model on a beefier box,
#    bump OLLAMA_VISION_MODEL to "llama3.2-vision:latest" (needs
#    11+ GB RAM free).
VISION_MODEL="${OLLAMA_VISION_MODEL:-moondream:latest}"
echo "📥 Pulling vision model: $VISION_MODEL ..."
ollama pull "$VISION_MODEL"

echo "✅ Models ready!"
echo "🎓 Smart Teacher is ready to use!"
