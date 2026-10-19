"""Together's response to transcript lines: pure logic with real responses' edge cases, so unit tests."""
import json
import pathlib

from agent.transcribe import transcript_lines

# a 29-second lesson as Together answered it, trimmed to the fields transcript_lines reads
DIALOG = pathlib.Path(__file__).parent.parent / "fixtures" / "together_dialog.json"


def test_a_segment_splits_where_the_speaker_changes():
    lines = transcript_lines(json.loads(DIALOG.read_text()))  # one 30-second segment, nine turns
    assert [(line["speaker"], line["text"]) for line in lines] == [
        ("1", "Good morning, class. Today we are talking about photosynthesis. What gas do plants give off?"),
        ("2", "Oxygen."), ("1", "Right. And what do they take in?"), ("2", "Carbon dioxide."),
        ("1", "Good. Where does it happen in the cell?"), ("2", "In the chloroplasts, I think."),
        ("1", "Exactly. Chloroplasts hold chlorophyll, which absorbs red and blue light. Any questions before we "
              "move on to the light reactions?"),
        ("3", "Yes. Why are leaves green then?"),  # Together's diarization heard a third voice
        ("1", "Because chlorophyll reflects green light instead of absorbing it.")]
    assert [line["start"] for line in lines] == sorted(line["start"] for line in lines)


def test_a_one_speaker_segment_keeps_whispers_text_and_times():
    result = {"segments": [{"start": 0.0, "end": 1.7, "text": " The code is 7."},
                           {"start": 1.7, "end": 3.0, "text": " ..."}],  # no words: no speaker
              "words": [{"word": "The", "start": 0.1, "end": 0.2, "speaker_id": "SPEAKER_04"},
                        {"word": "code", "start": 0.3, "end": 0.6, "speaker_id": "SPEAKER_04"}]}
    assert transcript_lines(result) == [{"start": 0.0, "end": 1.7, "speaker": "1", "text": "The code is 7."},
                                        {"start": 1.7, "end": 3.0, "speaker": None, "text": "..."}]
