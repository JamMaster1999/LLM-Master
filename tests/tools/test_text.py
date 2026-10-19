"""pi's read window and edit matching: pure logic with many edge cases, so unit tests."""
import pytest

from agent.files import Edit, apply_edits
from agent.reader import Problem, page
from agent.result import ToolError


def test_page_returns_a_short_file_whole():
    assert page("a\nb\n", "f.txt", None, None) == "a\nb"


def test_page_cuts_at_2000_lines_and_says_where_to_continue():
    out = page("\n".join(str(i) for i in range(1, 5001)), "f.txt", None, None)
    assert out.splitlines()[1999] == "2000"
    assert out.endswith("\n\n[Showing lines 1-2000 of 5000. Use offset=2001 to continue, or search f.txt with bash (rg, grep).]")


def test_page_cuts_at_50kb_between_lines():
    out = page("\n".join("x" * 99 for _ in range(1000)), "f.txt", 11, None)  # 100 bytes a line
    assert out.endswith("x\n\n[Showing lines 11-522 of 1000 (50.0KB limit). Use offset=523 to continue, "
                        "or search f.txt with bash (rg, grep).]")


def test_page_stops_at_the_limit_and_counts_whats_left():
    assert page("a\nb\nc\nd", "f.txt", 2, 2) == "b\nc\n\n[1 more lines in file. Use offset=4 to continue.]"


def test_page_rejects_an_offset_past_the_end():
    with pytest.raises(Problem, match=r"^Offset 9 is beyond end of file \(2 lines total\)$"):
        page("a\nb", "f.txt", 9, None)


def test_page_points_a_huge_line_at_bash():
    assert page("x" * 60_000, "f.txt", None, None) == (
        "[Line 1 is 58.6KB, exceeds 50.0KB limit. Use bash: sed -n '1p' f.txt | head -c 51200]")


def test_edits_match_the_original_and_keep_bom_and_line_endings():
    edits: list[Edit] = [{"oldText": "one", "newText": "two"}, {"oldText": "two\nthree", "newText": "2\n3"}]
    assert apply_edits("\ufeffone\r\ntwo\r\nthree\r\n", edits, "f") == "\ufefftwo\r\n2\r\n3\r\n"


@pytest.mark.parametrize("edits, message", [
    ([], "edits must contain at least one replacement"),
    ([{"oldText": "zzz", "newText": "y"}], "Could not find the exact text in f. "),
    ([{"oldText": "one", "newText": "1"}, {"oldText": "zzz", "newText": "y"}], r"Could not find edits\[1\] in f. "),
    ([{"oldText": "o", "newText": "0"}], "Found 2 occurrences of the text in f. "),
    ([{"oldText": "one t", "newText": "x"}, {"oldText": "e tw", "newText": "y"}], r"edits\[0\] and edits\[1\] overlap in f. "),
    ([{"oldText": "one", "newText": "one"}], "No changes made to f. "),
])
def test_edit_refusals_use_pis_wording(edits, message):
    with pytest.raises(ToolError, match=message):
        apply_edits("one two", edits, "f")


def test_overlapping_matches_count_as_two():
    with pytest.raises(ToolError, match="Found 2 occurrences of the text in f. "):
        apply_edits("banana", [{"oldText": "ana", "newText": "x"}], "f")
