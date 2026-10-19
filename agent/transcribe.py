"""Recordings' transcripts, from Together's Whisper with speakers. It runs on the host, since the
sandbox has no network."""
import os
from bisect import bisect_left
from itertools import groupby

import httpx

from agent.result import ToolError

TOGETHER_API = "https://api.together.ai/v1"


async def transcribe(name: str, data: bytes) -> tuple[float, list[dict]]:
    """The recording's length in seconds, and its lines as `{start, end, speaker, text}`, speakers
    numbered "1", "2", … in the order they first speak."""
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(
                f"{TOGETHER_API}/audio/transcriptions",
                headers={"Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}"},
                files={"file": (name, data)},  # Together reads the format from the bytes
                data={"model": "openai/whisper-large-v3", "response_format": "verbose_json",
                      "timestamp_granularities": "segment", "diarize": "true", "language": "auto"})
    except httpx.TransportError as error:  # a timeout or a dropped connection
        raise ToolError(f"Together couldn't transcribe {name} ({type(error).__name__}). Read it to try again.") from None
    if response.is_error:
        raise ToolError(f"Together couldn't transcribe {name} ({response.status_code} {response.text[:500]}). "
                        f"Read it to try again.")
    result = response.json()
    # A line is a Whisper segment, a few seconds long, split where the speaker changes: diarization
    # labels the words, and a student's short answer can share a segment with the teacher
    words = result.get("words") or []
    starts = [word["start"] for word in words]
    speakers: dict[str, str] = {}
    lines = []
    for segment in result["segments"]:
        said = words[bisect_left(starts, segment["start"]):bisect_left(starts, segment["end"])]
        turns = [list(turn) for _, turn in groupby(said, key=lambda word: word.get("speaker_id"))]
        if len(turns) < 2:  # one speaker: Whisper's own text and times
            turns = [[{"start": segment["start"], "end": segment["end"], "word": segment["text"],
                       "speaker_id": turns[0][0].get("speaker_id") if turns else None}]]
        for turn in turns:
            speaker = turn[0].get("speaker_id")
            lines.append({"start": turn[0]["start"], "end": turn[-1]["end"],
                          "speaker": speakers.setdefault(speaker, str(len(speakers) + 1)) if speaker else None,
                          "text": " ".join(word["word"].strip() for word in turn)})
    return result["duration"], lines
