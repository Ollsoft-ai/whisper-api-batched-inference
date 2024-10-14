from typing import Optional
import os
import time
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from contextlib import asynccontextmanager

from whisper_tools import WhisperTools

whisper_tools = WhisperTools()

@asynccontextmanager
async def lifespan(app: FastAPI):
    whisper_tools.setup()
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    num_speakers: Optional[int] = None,
    language: Optional[str] = None,
    prompt: str = ""
):
    temp_file = f"temp-{time.time_ns()}.wav"
    try:
        with open(temp_file, "wb") as buffer:
            buffer.write(await file.read())
        
        transcript, segments, detected_language, detected_num_speakers = whisper_tools.transcribe(
            temp_file,
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
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
