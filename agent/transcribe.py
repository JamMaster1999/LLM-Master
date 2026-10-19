"""Recordings' transcripts, from Together's Whisper with speakers. It runs on the host, since the
sandbox has no network."""
import os
from bisect import bisect_left
from itertools import groupby

import httpx
from typing_extensions import TypedDict

from agent.result import ToolError

TOGETHER_API = "https://api.together.ai/v1"


class Line(TypedDict):
    """One line of a recording's transcript.jsonl."""
    start: float
    end: float
    speaker: str | None  # "1", "2", … in the order they first speak
    text: str


async def transcribe(name: str, data: bytes) -> tuple[float, list[Line]]:
    """The recording's length in seconds, and its lines."""
    if not (key := os.environ.get("TOGETHER_API_KEY")):
        raise ToolError(f"{name} can't be transcribed: the host has no TOGETHER_API_KEY.")
    retry = "Read it to try again."
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(
                f"{TOGETHER_API}/audio/transcriptions", headers={"Authorization": f"Bearer {key}"},
                files={"file": (name, data)},  # Together reads the format from the bytes
                data={"model": "openai/whisper-large-v3", "response_format": "verbose_json",
                      "timestamp_granularities": ["segment", "word"], "diarize": "true", "language": "auto"})
    except httpx.TransportError as error:  # a timeout or a dropped connection
        raise ToolError(f"Together couldn't transcribe {name} ({type(error).__name__}). {retry}") from None
    if response.is_error:  # a 4xx is for good: too long, a codec it can't read, too big
        raise ToolError(f"Together couldn't transcribe {name} ({response.status_code} {response.text[:500]})."
                        + (f" {retry}" if response.is_server_error else ""))
    result = response.json()
    return result["duration"], transcript_lines(result)


def transcript_lines(result: dict) -> list[Line]:
    """Together's segments, each split where the speaker changes. A segment can run 30 seconds and
    hold a question and its answer; diarization labels the words, so the words show who said what."""
    words = result.get("words") or []
    starts = [word["start"] for word in words]
    speakers: dict[str, str] = {}

    def line(start: float, end: float, speaker: str | None, text: str) -> Line:
        return {"start": start, "end": end,
                "speaker": speakers.setdefault(speaker, str(len(speakers) + 1)) if speaker else None,
                "text": text.strip()}

    lines = []
    for segment in result["segments"]:
        said = words[bisect_left(starts, segment["start"]):bisect_left(starts, segment["end"])]
        turns = [list(turn) for _, turn in groupby(said, key=lambda word: word.get("speaker_id"))]
        if len(turns) < 2:  # one speaker: Whisper's own text and times
            lines.append(line(segment["start"], segment["end"], turns[0][0].get("speaker_id") if turns else None,
                              segment["text"]))
        else:
            lines += [line(turn[0]["start"], turn[-1]["end"], turn[0].get("speaker_id"),
                           " ".join(word["word"].strip() for word in turn)) for turn in turns]
    return lines
