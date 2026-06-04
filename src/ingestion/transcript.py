import logging
from dataclasses import dataclass
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import(
    TranscriptsDisabled,
    NoTranscriptFound
)

logger = logging.getLogger("rag_app.ingestion.transcript")

class TranscriptError(Exception):
    pass

class TranscriptNotAvailableError(TranscriptError):
    pass

class InvalidYoutubeURLError(TranscriptError):
    pass


@dataclass
class Transcript_segment:
    text : str
    start : float
    duration : float
    
    @property
    def end(self) -> float:
        return self.start + self.duration
    
    @property
    def timestamp(self) -> str:
        return _format_timestamp(self.start)
    
def _format_timestamp(seconds : float) -> str:
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) //60
    seconds = seconds % 60

    if hours>0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes}:{seconds:02d}"
    
class YoutubeTranscriptFetcher:

    def __init__(self):
        self._api = YouTubeTranscriptApi()
        logger.info("YouTubeTranscriptFetcher initialized")

    def extract_video_id(self,url:str) -> str:

        url = url.strip()

        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1]  
        else:
            logger.warning(f"Unrecognized YouTube URL format:{url}")
            raise InvalidYoutubeURLError(
                f"Could not extract video ID from URL. "
                f"Expected format: youtube.com/watch?v=ID or youtu.be/ID"
            )
        logger.debug(f"extracted Youtube ID:{video_id}")
        return video_id
    
    def fetch(self,url:str) -> tuple[str,list[Transcript_segment]]:
        logger.info(f"Fetching Transcript for:{url}")

        video_id = self.extract_video_id(url)
        try:
            raw_transcript = self._api.fetch(video_id=video_id)
        except TranscriptsDisabled:
            logger.warning(f"Trascript for this video is disabled : {video_id}")
            raise TranscriptNotAvailableError(
                f"The owner of video '{video_id}' has disabled transcripts."
            )
        except NoTranscriptFound:
            logger.warning(f"transcript not available for this vide : {video_id}")
            raise TranscriptNotAvailableError(
                f"No transcript available for video '{video_id}'. "
                f"This may be a live stream or a video without captions."
            )
        
        segments = [
            Transcript_segment(
                text = seg.text.strip(),
                start = seg.start,
                duration = seg.duration
            )
            for seg in raw_transcript
        ]

        total_duration = segments[-1].end if segments else 0
        logger.info(
            f"Transcript Fetched | video ID = {video_id}"
            f"segment= {len(segments)}"
            f"total duration = {_format_timestamp(total_duration)}"
        )
        
        return video_id, segments
    
__all__ = [
    YoutubeTranscriptFetcher,
    Transcript_segment,
    InvalidYoutubeURLError,
    TranscriptError,
    TranscriptNotAvailableError
]    

            
           
    

    