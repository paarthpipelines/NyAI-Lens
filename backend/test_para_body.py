import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import para_body


def test_plain_prose_soft_wrap():
    chunk = "1. This is the first line\nand this continues the sentence naturally"
    result = para_body(chunk)
    assert "first line and this continues" in result


def test_italic_quoted_letter_not_joined():
    chunk = (
        "1. Some prose before the letter.\n"
        "_\"I am directed to state that the Tamil Nadu Fisheries University\n"
        "(Amendment) Bill, 2023 was passed on 21.4.2023 and sent to the\n"
        "Hon'ble Governor for assent.\"_"
    )
    result = para_body(chunk)
    lines = result.splitlines()
    # Italic block must not be joined onto the prose line
    assert not any(
        "Some prose" in l and "directed" in l for l in lines
    ), "Prose line should not be joined with the italic quoted letter"
    # At least one italic line must be preserved
    italic_lines = [l for l in lines if l.strip().startswith('_')]
    assert len(italic_lines) >= 1, "Italic lines should remain on their own lines"


def test_bold_inline_lowercase_not_joined():
    chunk = "1. The court considered the matter.\n**held** that the act was valid."
    result = para_body(chunk)
    lines = result.splitlines()
    assert any(l.strip().startswith("**held**") for l in lines), \
        "**held** line should remain on its own line"


def test_heading_preserved():
    chunk = "1. Some introductory text.\n## UNION LIST\nMore text after heading."
    result = para_body(chunk)
    lines = result.splitlines()
    assert any(l.strip() == "## UNION LIST" for l in lines), \
        "## heading must not be joined onto previous line"


def test_footnote_reference_not_joined():
    chunk = "1. The court cited precedent.\n[1] AIR 1950 SC 27."
    result = para_body(chunk)
    lines = result.splitlines()
    assert any(l.strip().startswith("[1]") for l in lines), \
        "Footnote [1] line should remain on its own line"


def test_blank_line_preserved():
    chunk = (
        "1. _\"First quoted block.\"_\n"
        "\n"
        "_\"Second quoted block.\"_"
    )
    result = para_body(chunk)
    lines = result.splitlines()
    blank_lines = [l for l in lines if l.strip() == ""]
    assert len(blank_lines) >= 1, "Blank line between italic blocks must be preserved"
    italic_lines = [l for l in lines if l.strip().startswith("_")]
    assert len(italic_lines) >= 2, "Both italic blocks should appear as separate lines"


def test_mixed_paragraph():
    body = (
        "On 28.11.2023, the Governor reserved the Bills for consideration.\n"
        "Interestingly the Bills were _**intra-vires**_ the State Legislature.\n"
        "\n"
        "_\"I am directed to state that the Bill was passed on 21.4.2023\n"
        "and sent to the Hon'ble Governor for assent. Hon'ble Governor\n"
        "returned the Bill with the following remarks:\"\n"
        "\n"
        "_\"I withhold assent\"._\n"
        "\n"
        "## UNION LIST\n"
        "\n"
        "_Entry 66 - Co-ordination and determination of standards_"
    )
    # wrap in a para chunk so para_body strips the numbering prefix cleanly
    chunk = "1. " + body
    result = para_body(chunk)
    lines = result.splitlines()

    assert any(l.strip() == "## UNION LIST" for l in lines), \
        "## UNION LIST heading must appear on its own line"

    assert any(l.strip() == '_"I withhold assent"._' for l in lines), \
        '_"I withhold assent"._ must appear on its own line'

    # No italic lines should have been merged into a single flat non-italic string
    for l in lines:
        if "_" not in l and "italic" not in l:
            assert "_\"I am directed" not in l, \
                "Italic quoted letter lines must not appear in a non-italic merged line"
