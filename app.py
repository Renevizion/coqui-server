import os
import re
import tempfile
from typing import Optional

import aiofiles
import demoji
import pyflac
import torch
from fastapi import FastAPI, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from TTS.api import TTS

STYLES = [
    {"style": "BOLD", "regex": r"\*\*(.*?)\*\*", "offset": 2},
    {"style": "ITALIC", "regex": r"\*(.*?)\*", "offset": 1}
]

app = FastAPI()

origins = ["*", "http://localhost:8000", "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
tts_client = TTS(
    model_name="tts_models/en/vctk/vits",
    progress_bar=False
).to(device)


def remove_markdown_styles(text: str):
    message = str(text)
    for style in STYLES:
        regex = style["regex"]
        offset = style["offset"]
        match = re.search(regex, message)

        while match:
            group = match.group()
            message = message.replace(group, group[offset:-offset], 1)
            match = re.search(regex, message)

    return message


def clean_text_for_tts(text):
    """Cleans text for better TTS output."""
    text = os.linesep.join([s for s in text.splitlines() if s])
    text = demoji.replace(text, "")
    text = remove_markdown_styles(text)

    text = text.replace("%", " percent")
    text = text.replace("*", "-")
    text = text.replace("  +", "  -")
    text = text.replace("\r", "").strip()
    text = re.sub(" +", " ", text)

    text = text.replace("°F", "° Fahrenheit")
    text = text.replace("°C", "° Celsius")
    text = text.replace("°K", "° Kelvin")

    return text


@app.get("/")
async def read_root():
    return {"device": device}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/tts")
async def tts(
    text: str = Form(None, description="Text to convert to speech."),
    speaker_id: int = Form(None, description="VCTK speaker as an integer to use. Note that they're shuffled from the original dataset in Coqui."),
    speed: Optional[float] = Form(1.0, description="Controls the playback speed. Available to tune since some speakers can be VERY fast."),
    compress: Optional[bool] = Form(True, description="Compress the audio into FLAC.")
):
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text to convert into speech must be provided.",
        )

    if not speaker_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Speaker ID must be provided.",
        )

    # Create a real temp WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name

    # Generate TTS and save to WAV path
    tts_client.tts_to_file(
        text=clean_text_for_tts(text),
        speaker=f"p{speaker_id}",
        speed=speed,
        file_path=wav_path
    )

    if compress:
        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp_flac:
            flac_path = tmp_flac.name

        # Compress using pyflac
        encoder = pyflac.FileEncoder(input_file=wav_path, output_file=flac_path)
        encoder.process()
        encoder.finish()

        async with aiofiles.open(flac_path, "rb") as flac:
            content = await flac.read()

        os.remove(wav_path)
        os.remove(flac_path)
        return Response(content=content, media_type="audio/flac")

    # If not compressing, just return the WAV file
    async with aiofiles.open(wav_path, "rb") as audio:
        content = await audio.read()

    os.remove(wav_path)
    return Response(content=content, media_type="audio/wav")
