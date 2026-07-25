import pytest
from src.core.transcript import extract_video_id


def test_extract_video_id_standard_url():
    url = "https://www.youtube.com/watch?v=EV7sYWGKAVU"
    assert extract_video_id(url) == "EV7sYWGKAVU"


def test_extract_video_id_with_extra_params():
    url = "https://www.youtube.com/watch?v=EV7sYWGKAVU&t=42s&list=PL123"
    assert extract_video_id(url) == "EV7sYWGKAVU"


def test_extract_video_id_short_url():
    url = "https://youtu.be/EV7sYWGKAVU"
    assert extract_video_id(url) == "EV7sYWGKAVU"


def test_extract_video_id_short_url_with_params():
    url = "https://youtu.be/EV7sYWGKAVU?si=abc123"
    assert extract_video_id(url) == "EV7sYWGKAVU"


def test_extract_video_id_empty_raises():
    with pytest.raises(ValueError):
        extract_video_id("")


def test_extract_video_id_invalid_raises():
    with pytest.raises(ValueError):
        extract_video_id("https://example.com/not-a-video")