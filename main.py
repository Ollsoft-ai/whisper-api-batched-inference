from typing import Optional
import os
import time
from fastapi import FastAPI, File, UploadFile, Form, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
from tempfile import NamedTemporaryFile
import logging

from whisper_tools import WhisperTools

whisper_tools = WhisperTools()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    whisper_tools.setup()
    yield

app = FastAPI(lifespan=lifespan)

# Add health check endpoint
@app.get("/health")
async def health_check():
    try:
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "service": "whisper-api"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time(),
            "service": "whisper-api"
        }, 503

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    num_speakers: Optional[int] = None,
    language: Optional[str] = None,
    prompt: str = ""
):
    try:
        with NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            transcript, segments, detected_language, detected_num_speakers = whisper_tools.transcribe(
                temp_file.name,
                num_speakers,
                language,
                prompt
            )
            
            return {
                "transcript": transcript,
                "segments": segments,
                "detected_language": detected_language,
                "detected_num_speakers": detected_num_speakers
            }
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
