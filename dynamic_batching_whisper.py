# app.py
import torch
from fastapi import FastAPI, UploadFile, BackgroundTasks
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import List, Optional
import numpy as np
import soundfile as sf
import io
from datetime import datetime
import asyncio
from pydantic import BaseModel

class WhisperBatchInference:
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        batch_size: int = 24,
        max_wait_ms: int = 10000,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.pending_batch = []
        self.batch_lock = asyncio.Lock()
        self.setup_model()

    def setup_model(self):
        if (not torch.cuda.is_available()):
            raise Exception("CUDA is not available")
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to("cuda")

        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch.float16,
            device="cuda",
        )

    async def process_audio(self, audio_data: np.ndarray) -> str:
        async with self.batch_lock:
            # Add current request to pending batch
            future = asyncio.Future()
            self.pending_batch.append((audio_data, future))
            
            # If batch is full, process immediately
            if len(self.pending_batch) >= self.batch_size:
                await self.process_batch()
            else:
                # Schedule batch processing after wait time
                asyncio.create_task(self.schedule_batch_processing())

        # Wait for result
        result = await future
        return result

    async def schedule_batch_processing(self):
        await asyncio.sleep(self.max_wait_ms / 1000)  # Convert ms to seconds
        async with self.batch_lock:
            if self.pending_batch:  # Check if there are still pending items
                await self.process_batch()

    async def process_batch(self):
        # Separate audio data and futures
        audio_batch, futures = zip(*self.pending_batch)
        self.pending_batch.clear()

        # Process batch
        try:
            results = self.pipeline(list(audio_batch), batch_size=len(audio_batch))
            # Set results for each future
            for future, result in zip(futures, results):
                future.set_result(result["text"])
        except Exception as e:
            # Handle errors
            for future in futures:
                if not future.done():
                    future.set_exception(e)

app = FastAPI()
whisper = WhisperBatchInference()

class TranscriptionResponse(BaseModel):
    text: str
    processing_time: float

@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile):
    start_time = datetime.now()
    
    # Read and process audio file
    content = await file.read()
    audio_data, _ = sf.read(io.BytesIO(content))
    
    # Get transcription
    transcription = await whisper.process_audio(audio_data)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return TranscriptionResponse(
        text=transcription,
        processing_time=processing_time
    )

@app.post("/transcribe_batch/", response_model=List[TranscriptionResponse])
async def transcribe_batch(files: List[UploadFile]):
    start_time = datetime.now()
    
    # Process all files in parallel
    async def process_file(file):
        content = await file.read()
        audio_data, _ = sf.read(io.BytesIO(content))
        return await whisper.process_audio(audio_data)
    
    transcriptions = await asyncio.gather(
        *[process_file(file) for file in files]
    )
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return [
        TranscriptionResponse(text=text, processing_time=processing_time)
        for text in transcriptions
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)