import requests
import os
from dataclasses import dataclass


@dataclass
class ApiSource:
    rank        : int
    start_time  : str
    end_time    : str
    youtube_link: str
    display     : str


@dataclass
class ApiResponse:
    answer            : str
    sources           : list[ApiSource]
    queries_used      : list[str]
    video_id          : str
    answer_grounded   : bool
    ingestion_skipped : bool
    total_ms          : float

class RAGApiClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (
            base_url
            or os.getenv("FASTAPI_URL", "http://localhost:8000")
        ).rstrip("/")

        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
        })

    def health(self) -> bool:
        try:
            resp = self.session.get(
                f"{self.base_url}/health",
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def is_video_indexed(self, youtube_url: str) -> bool:
        try:
            video_id = self._extract_video_id(youtube_url)
            resp     = self.session.get(
                f"{self.base_url}/videos",
                timeout=10,
            )
            data = resp.json()
            return video_id in data.get("video_ids", [])
        except Exception:
            return False

    def ingest(self, youtube_url: str) -> dict:
        resp = self.session.post(
            f"{self.base_url}/ingest",
            json    = {"video_url": youtube_url},
            timeout = 120,   
        )
        resp.raise_for_status()
        return resp.json()

    def chat(
        self,
        youtube_url : str,
        question    : str,
        history     : list[dict] | None = None,
    ) -> ApiResponse:
        resp = self.session.post(
            f"{self.base_url}/chat",
            json = {
                "video_url": youtube_url,
                "question" : question,
                "history"  : history or [],
            },
            timeout = 60,
        )
        resp.raise_for_status()
        data = resp.json()

        sources = [
            ApiSource(
                rank         = s["rank"],
                start_time   = s["start_time"],
                end_time     = s["end_time"],
                youtube_link = s["youtube_link"],
                display      = s["display"],
            )
            for s in data.get("sources", [])
        ]

        return ApiResponse(
            answer            = data["answer"],
            sources           = sources,
            queries_used      = data.get("queries_used", []),
            video_id          = data["video_id"],
            answer_grounded   = data.get("answer_grounded", True),
            ingestion_skipped = data.get("ingestion_skipped", False),
            total_ms          = data.get("latency_ms", {}).get("total", 0),
        )

    def _extract_video_id(self, url: str) -> str:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(url)
        if "v" in parse_qs(parsed.query):
            return parse_qs(parsed.query)["v"][0]
        if "youtu.be" in parsed.netloc:
            return parsed.path.lstrip("/")
        return ""    