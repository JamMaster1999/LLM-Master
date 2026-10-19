"""Recordings' transcripts, from Together's Whisper with speakers. It runs on the host, since the
sandbox has no network."""
import os
from bisect import bisect_left

import httpx

from agent.result import ToolError

TOGETHER_API = "https://api.together.ai/v1"


async def transcribe(name: str, data: bytes) -> list[dict]:
    """The recording's segments as `{start, end, speaker, text}`, speakers numbered "1", "2", … in the
    order they first speak."""
    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(
            f"{TOGETHER_API}/audio/transcriptions", headers={"Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}"},
            files={"file": (name, data)},  # Together reads the format from the bytes
            data={"model": "openai/whisper-large-v3", "response_format": "verbose_json",
                  "timestamp_granularities": "segment", "diarize": "true", "language": "auto"})
    if response.is_error:
        raise ToolError(f"Together couldn't transcribe {name}: {response.status_code} {response.text[:500]}")
    result = response.json()
    # Whisper's segments are a few seconds each; diarization labels the words, so each segment
    # takes the speaker of most of its words
    words = result.get("words") or []
    starts = [word["start"] for word in words]
    speakers: dict[str, str] = {}
    segments = []
    for segment in result["segments"]:
        said = [word["speaker_id"] for word in words[bisect_left(starts, segment["start"]):bisect_left(starts, segment["end"])]
                if "speaker_id" in word]
        speaker = speakers.setdefault(max(said, key=said.count), str(len(speakers) + 1)) if said else None
        segments.append({"start": segment["start"], "end": segment["end"], "speaker": speaker,
                         "text": segment["text"].strip()})
    return segments
