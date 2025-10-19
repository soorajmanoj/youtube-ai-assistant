from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    raise ValueError("Invalid YouTube URL")

def get_transcript(url: str) -> str:
    video_id = extract_video_id(url)
    data = YouTubeTranscriptApi().fetch(video_id)
    return " ".join([snippet.text for snippet in data])
