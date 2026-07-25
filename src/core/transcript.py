from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)


def extract_video_id(url: str) -> str:
    if not url or not url.strip():
        raise ValueError("No URL provided.")
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    raise ValueError(
        "Invalid YouTube URL. Expected something like "
        "https://www.youtube.com/watch?v=... or https://youtu.be/..."
    )


def get_transcript(url: str) -> str:
    video_id = extract_video_id(url)

    try:
        data = YouTubeTranscriptApi().fetch(video_id)
    except TranscriptsDisabled:
        raise ValueError("This video has captions disabled, so no transcript is available.")
    except NoTranscriptFound:
        raise ValueError("No transcript could be found for this video.")
    except VideoUnavailable:
        raise ValueError("This video is unavailable (private, deleted, or region-locked).")
    except Exception as e:
        raise ValueError(f"Failed to fetch transcript: {e}")

    if not data:
        raise ValueError("Transcript was empty.")

    return " ".join([snippet.text for snippet in data])