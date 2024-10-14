from typing import Any, List, Optional
import datetime
import subprocess
import os
import time
import torch
import numpy as np
from transformers import pipeline
from pyannote.audio import Pipeline
import torchaudio
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn

class WhisperTools():
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("cudaa", torch.cuda.is_available(), torch.version.cuda)

        model_name = os.getenv("ASR_MODEL", "openai/whisper-large-v3")
        self.model = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=torch.float16,
            device="cuda",
            model_kwargs={"attn_implementation": "sdpa"},
        )
        print(f"loaded model: {model_name}")

        self.diarization_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.getenv("HF_TOKEN"),
            cache_dir=os.getenv("HUGGINGFACE_HUB_CACHE", "models"),
        ).to(torch.device("cuda"))
        print(f"loaded diarization model")
        
    def transcribe(
        self,
        file: str,
        num_speakers: Optional[int] = None,
        language: Optional[str] = None,
        prompt: str = "",
    ):
        try:
            # Generate a temporary filename
            temp_wav_filename = f"temp-{time.time_ns()}.wav"

            # Convert the audio file to a WAV file
            if file is not None:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        file,
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_s16le",
                        temp_wav_filename,
                    ]
                )

            transcript, segments, detected_num_speakers, detected_language = self.speech_to_text(
                temp_wav_filename,
                num_speakers,
                prompt=prompt,
                language=language,
            )

            print(f"done with inference")
            # Return the results as a JSON object
            return transcript, segments, detected_language, detected_num_speakers

        except Exception as e:
            raise RuntimeError("Error Running inference with local model", e)

        finally:
            # Clean up
            if os.path.exists(temp_wav_filename):
                os.remove(temp_wav_filename)

    def convert_time(self, secs, offset_seconds=0):
        return datetime.timedelta(seconds=(round(secs) + offset_seconds))

    def speech_to_text(
        self,
        audio_file_wav,
        num_speakers=None,
        prompt="",
        offset_seconds=0,
        group_segments=True,
        language=None,
        translate=False,
    ):
        time_start = time.time()

        # Transcribe audio
        print("Starting transcribing")
        generate_kwargs = {
            "task": "translate" if translate else "transcribe",
            "language": language if language != "None" else None,
        }

        with Progress(
            TextColumn("🤗 [progress.description]{task.description}"),
            BarColumn(style="yellow1", pulse_style="white"),
            TimeElapsedColumn(),
        ) as progress:
            progress.add_task("[yellow]Transcribing...", total=None)

            outputs = self.model(
                audio_file_wav,
                chunk_length_s=30,
                batch_size=24,
                generate_kwargs=generate_kwargs,
                return_timestamps=True,
            )

        segments = outputs["chunks"]
        
        time_transcribing_end = time.time()
        print(
            f"Finished with transcribing, took {time_transcribing_end - time_start:.5} seconds"
        )

        print("Starting diarization")
        waveform, sample_rate = torchaudio.load(audio_file_wav)
        diarization = self.diarization_model(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=num_speakers,
        )

        time_diraization_end = time.time()
        print(
            f"Finished with diarization, took {time_diraization_end - time_transcribing_end:.5} seconds"
        )

        print("Starting merging")

        # Initialize variables to keep track of the current position in both lists
        margin = 0.1  # 0.1 seconds margin

        # Initialize an empty list to hold the final segments with speaker info
        final_segments = []

        diarization_list = list(diarization.itertracks(yield_label=True))
        unique_speakers = {
            speaker for _, _, speaker in diarization.itertracks(yield_label=True)
        }
        detected_num_speakers = len(unique_speakers)

        speaker_idx = 0
        n_speakers = len(diarization_list)

        # Iterate over each segment
        for segment in segments:
            segment_start = segment["timestamp"][0] + offset_seconds
            segment_end = segment["timestamp"][1] + offset_seconds
            segment_text = segment["text"]

            while speaker_idx < n_speakers:
                turn, _, speaker = diarization_list[speaker_idx]

                if turn.start <= segment_end and turn.end >= segment_start:
                    new_segment = {
                        "start": segment_start - offset_seconds,
                        "end": segment_end - offset_seconds,
                        "speaker": speaker,
                        "text": segment_text,
                    }
                    final_segments.append(new_segment)

                    if turn.end <= segment_end:
                        speaker_idx += 1
                    break
                elif turn.end < segment_start:
                    speaker_idx += 1
                else:
                    break

        time_merging_end = time.time()
        print(
            f"Finished with merging, took {time_merging_end - time_diraization_end:.5} seconds"
        )

        print("Starting cleaning")
        segments = final_segments
        # Make output
        output = []  # Initialize an empty list for the output

        # Initialize the first group with the first segment
        current_group = {
            "start": segments[0]["start"],
            "end": segments[0]["end"],
            "speaker": segments[0]["speaker"],
            "text": segments[0]["text"],
        }

        for i in range(1, len(segments)):
            # Calculate time gap between consecutive segments
            time_gap = segments[i]["start"] - segments[i - 1]["end"]

            # If the current segment's speaker is the same as the previous segment's speaker,
            # and the time gap is less than or equal to 2 seconds, group them
            if segments[i]["speaker"] == segments[i - 1]["speaker"] and time_gap <= 2 and group_segments:
                current_group["end"] = segments[i]["end"]
                current_group["text"] += " " + segments[i]["text"]
            else:
                # Add the current_group to the output list
                output.append(current_group)

                # Start a new group with the current segment
                current_group = {
                    "start": segments[i]["start"],
                    "end": segments[i]["end"],
                    "speaker": segments[i]["speaker"],
                    "text": segments[i]["text"],
                }

        # Create transcript string
        transcript = ""
        previous_speaker = None
        combined_text = ""

        for segment in segments:
            speaker = segment['speaker']
            text = segment['text']
            
            if speaker == previous_speaker:
                combined_text += f" {text}"
            else:
                if previous_speaker is not None:
                    transcript += f"{previous_speaker}: {combined_text}\n\n"
                combined_text = text
                previous_speaker = speaker

        # Add the last segment
        if previous_speaker is not None:
            transcript += f"{previous_speaker}: {combined_text}\n\n"

        transcript = transcript.strip()
        
        # Add the last group to the output list
        output.append(current_group)

        time_cleaning_end = time.time()
        print(
            f"Finished with cleaning, took {time_cleaning_end - time_merging_end:.5} seconds"
        )
        time_end = time.time()
        time_diff = time_end - time_start

        system_info = f"""Processing time: {time_diff:.5} seconds"""
        print(system_info)
        return transcript, output, detected_num_speakers, "language"
