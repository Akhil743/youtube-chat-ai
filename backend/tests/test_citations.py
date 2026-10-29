import pytest

from app.services.citations import (
    format_chunk, format_range, format_ts, is_refusal, parse_ts, validate_citations,
)

SPANS = [(12.4, 41.2), (65.0, 92.7)]  # displayed as [00:12–00:42] and [01:05–01:33]


@pytest.mark.parametrize("secs,label", [(0, "00:00"), (5.9, "00:05"), (83, "01:23"), (599, "09:59"), (3600, "1:00:00"), (3725.2, "1:02:05")])
def test_format_ts(secs, label):
    assert format_ts(secs) == label


@pytest.mark.parametrize("label,secs", [("00:00", 0), ("1:23", 83), ("01:23", 83), ("75:10", 4510), ("1:02:05", 3725)])
def test_parse_ts(label, secs):
    assert parse_ts(label) == secs


@pytest.mark.parametrize("bad", ["", "1:2:3:4", "aa:bb", "01:60", "1:61:00", "123"])
def test_parse_ts_rejects(bad):
    with pytest.raises(ValueError):
        parse_ts(bad)


def test_chunk_header_floors_start_and_ceils_end():
    assert format_range(12.4, 41.2) == "00:12–00:42"
    assert format_chunk("hello", 12.4, 41.2) == "[00:12–00:42] hello"


def test_valid_citations_kept_and_normalised():
    check = validate_citations("Ranking uses links [00:12] and freshness [1:05].", SPANS)
    assert check.answer == "Ranking uses links [00:12] and freshness [01:05]."
    assert check.cited == [12, 65] and check.valid == [12, 65] and check.removed == []
    assert check.validity == 1.0


def test_boundaries_are_inclusive_of_displayed_range():
    check = validate_citations("a [00:42] b [01:33]", SPANS)
    assert check.valid == [42, 93]


def test_invented_timestamp_removed():
    check = validate_citations("It starts [00:12]. Later it says more [03:30].", SPANS)
    assert check.answer == "It starts [00:12]. Later it says more."
    assert check.cited == [12, 210] and check.valid == [12]
    assert check.removed == ["[03:30]"]
    assert check.validity == 0.5


def test_range_kept_only_if_both_ends_supported():
    ok = validate_citations("See [00:12–00:42].", SPANS)
    assert ok.answer == "See [00:12–00:42]." and not ok.removed
    bad = validate_citations("See [00:12 - 02:00].", SPANS)
    assert bad.answer == "See." and bad.removed == ["[00:12 - 02:00]"]


def test_list_citation_keeps_supported_members():
    check = validate_citations("Both [00:20, 05:00].", SPANS)
    assert check.answer == "Both [00:20]."
    assert check.removed == ["[00:20, 05:00]"]


def test_parenthesised_citation_is_checked_too():
    check = validate_citations("Spam is filtered (09:09).", SPANS)
    assert check.answer == "Spam is filtered." and check.valid == []


def test_no_citations():
    check = validate_citations("No timestamps here, at 10:30 am.", SPANS)
    assert check.answer == "No timestamps here, at 10:30 am."
    assert check.cited == [] and check.validity is None


def test_refusal_detection():
    assert is_refusal("I couldn't find that in this video's transcript.")
    assert is_refusal("Sorry — I could not find that in this video.")
    assert not is_refusal("The video says lead sulfate builds up [02:41].")
