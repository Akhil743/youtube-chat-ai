import pytest

from app.services.transcript import extract_video_id

VID = "dQw4w9WgXcQ"


@pytest.mark.parametrize("url", [
    f"https://www.youtube.com/watch?v={VID}",
    f"https://youtube.com/watch?v={VID}&t=42s",
    f"https://m.youtube.com/watch?feature=share&v={VID}",
    f"https://music.youtube.com/watch?v={VID}&list=RD",
    f"www.youtube.com/watch?v={VID}",
    f"https://youtu.be/{VID}",
    f"https://youtu.be/{VID}?si=abc&t=10",
    f"https://www.youtube.com/embed/{VID}?start=5",
    f"https://www.youtube-nocookie.com/embed/{VID}",
    f"https://www.youtube.com/shorts/{VID}",
    f"https://www.youtube.com/live/{VID}?feature=share",
    f"https://www.youtube.com/v/{VID}",
    VID,
    f"  {VID}  ",
])
def test_extracts_id(url):
    assert extract_video_id(url) == VID


@pytest.mark.parametrize("url", [
    "",
    "not a url",
    "https://www.youtube.com/",
    "https://www.youtube.com/watch?v=short",
    "https://www.youtube.com/channel/UCabcdefghijk",
    f"https://evil.example.com/watch?v={VID}",
    f"https://vimeo.com/{VID}",
    "https://youtu.be/",
])
def test_rejects_non_video_urls(url):
    with pytest.raises(ValueError):
        extract_video_id(url)
