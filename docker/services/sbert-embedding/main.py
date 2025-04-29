import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
import numpy as np
import logging
import onnxruntime # Import onnxruntime to check providers

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# You can make the model name configurable via environment variable
MODEL_NAME = os.getenv("SBERT_MODEL_NAME", "all-MiniLM-L6-v2")
# Determine device based on ONNX Runtime provider availability (can be refined in Dockerfile)
# Defaulting to CPU here, ONNX Runtime will use available providers if installed
DEVICE = os.getenv("SBERT_DEVICE", "cpu") # e.g., 'cpu', 'cuda'

# --- Model Loading ---
model = None

def load_model():
    global model
    logger.info(f"Loading Sentence Transformer model: {MODEL_NAME} on device: {DEVICE}")
    start_time = time.time()
    try:
        model = SentenceTransformer(MODEL_NAME, device=DEVICE, backend="onnx")
        # Log available ONNX Runtime providers for confirmation
        logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds.")
        try:
            providers = onnxruntime.get_available_providers()
            logger.info(f"Available ONNX Runtime providers: {providers}")
            if "CPUExecutionProvider" not in providers:
                 logger.warning("CPUExecutionProvider not found in ONNX Runtime providers!")
        except Exception as ort_e:
            logger.warning(f"Could not get ONNX Runtime providers: {ort_e}")
    except Exception as e:
        logger.error(f"Error loading model {MODEL_NAME}: {e}", exc_info=True)
        # Depending on the desired behavior, you might want the app to fail startup
        raise RuntimeError(f"Failed to load model {MODEL_NAME}") from e

# --- Pydantic Models for OpenAI Compatibility ---
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = MODEL_NAME # Model name is often passed, though we use our configured one
    # encoding_format: Optional[str] = "float" # Optional parameter
    # user: Optional[str] = None # Optional parameter

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class Usage(BaseModel):
    prompt_tokens: int = 0 # SBERT token count is different from LLMs, often set to 0
    total_tokens: int = 0

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str = MODEL_NAME # Return the loaded model name
    usage: Usage

# --- FastAPI App ---
app = FastAPI(title="Sentence Transformer Embeddings", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Generates embeddings for the input text(s) using the loaded Sentence Transformer model.
    Compatible with the OpenAI Embeddings API format.
    """
    if model is None:
        logger.error("Model not loaded. Service cannot process requests.")
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")

    try:
        start_time = time.time()
        texts = [request.input] if isinstance(request.input, str) else request.input
        logger.info(f"Received request to embed {len(texts)} text(s). First text: '{texts[0][:80]}...'")

        # Generate embeddings
        # The encode method handles tokenization and embedding generation
        embeddings = model.encode(texts, convert_to_numpy=True)

        # Ensure embeddings are lists of floats
        embeddings_list = [emb.tolist() for emb in embeddings]

        # Prepare response data
        response_data: List[EmbeddingData] = []
        for i, emb_list in enumerate(embeddings_list):
            response_data.append(EmbeddingData(embedding=emb_list, index=i))

        # Calculate usage (placeholder for SBERT)
        # You could try to estimate based on model's tokenizer if needed, but it's non-standard
        usage = Usage(prompt_tokens=0, total_tokens=0)

        end_time = time.time()
        logger.info(f"Embeddings generated in {end_time - start_time:.4f} seconds.")

        return EmbeddingResponse(data=response_data, model=MODEL_NAME, usage=usage)

    except Exception as e:
        logger.error(f"Error processing embedding request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_name": MODEL_NAME}

# --- Run with Uvicorn (for local testing) ---
# You would typically run this using: uvicorn main:app --host 0.0.0.0 --port 8000
# The Dockerfile will handle running uvicorn.
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local testing...")
    # Load model immediately for local run
    if model is None:
        load_model()
    # Note: Use reload=True only for development
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
