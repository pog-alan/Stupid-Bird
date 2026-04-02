from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, List, Optional
from urllib import parse, request, robotparser


@dataclass(frozen=True)
class CrawlSource:
    name: str
    seed_urls: tuple[str, ...]
    allowed_domains: tuple[str, ...]
    max_depth: int = 1
    rate_limit_seconds: float = 1.0
    user_agent: str = "SB-V01-Crawler/0.1"


@dataclass
class CrawlerConfig:
    sources: tuple[CrawlSource, ...]
    max_pages: int = 200
    timeout_seconds: int = 10
    allow_robots: bool = True
    allow_redirects: bool = True
    accepted_mime_prefixes: tuple[str, ...] = ("text/", "application/json")


@dataclass
class CrawlResult:
    url: str
    status: int
    content_type: str
    text: str
    fetched_at: str
    error: Optional[str] = None


class Crawler:
    def __init__(self, config: CrawlerConfig) -> None:
        self.config = config
        self._robot_cache: dict[str, robotparser.RobotFileParser] = {}

    def crawl(self) -> List[CrawlResult]:
        results: List[CrawlResult] = []
        visited: set[str] = set()

        for source in self.config.sources:
            queue: List[tuple[str, int]] = [(url, 0) for url in source.seed_urls]
            while queue and len(results) < self.config.max_pages:
                url, depth = queue.pop(0)
                if url in visited or depth > source.max_depth:
                    continue
                if not self._is_allowed_domain(url, source.allowed_domains):
                    continue
                visited.add(url)

                if self.config.allow_robots and not self._allowed_by_robots(url, source.user_agent):
                    continue

                result = self._fetch(url, source.user_agent)
                results.append(result)
                if result.error:
                    continue

                if depth < source.max_depth:
                    for link in _extract_links(result.text, url):
                        if link not in visited:
                            queue.append((link, depth + 1))
        return results

    def _fetch(self, url: str, user_agent: str) -> CrawlResult:
        headers = {"User-Agent": user_agent}
        req = request.Request(url, headers=headers)
        fetched_at = datetime.now(timezone.utc).isoformat()
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                status = response.getcode() or 0
                content_type = response.headers.get("Content-Type", "")
                if not self._accepts_content_type(content_type):
                    return CrawlResult(
                        url=url,
                        status=status,
                        content_type=content_type,
                        text="",
                        fetched_at=fetched_at,
                        error="Unsupported content type",
                    )
                raw = response.read()
                encoding = response.headers.get_content_charset() or "utf-8"
                text = raw.decode(encoding, errors="replace")
                return CrawlResult(
                    url=url,
                    status=status,
                    content_type=content_type,
                    text=text,
                    fetched_at=fetched_at,
                )
        except Exception as exc:  # pragma: no cover - network errors are environment dependent
            return CrawlResult(
                url=url,
                status=0,
                content_type="",
                text="",
                fetched_at=fetched_at,
                error=str(exc),
            )

    def _accepts_content_type(self, content_type: str) -> bool:
        normalized = content_type.lower()
        for prefix in self.config.accepted_mime_prefixes:
            if normalized.startswith(prefix):
                return True
        return False

    def _is_allowed_domain(self, url: str, allowed_domains: Iterable[str]) -> bool:
        domain = parse.urlparse(url).netloc.lower()
        return any(domain.endswith(allowed.lower()) for allowed in allowed_domains)

    def _allowed_by_robots(self, url: str, user_agent: str) -> bool:
        parsed = parse.urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robot_cache:
            rp = robotparser.RobotFileParser()
            rp.set_url(parse.urljoin(base, "/robots.txt"))
            try:
                rp.read()
            except Exception:
                rp = robotparser.RobotFileParser()
                rp.parse([])
            self._robot_cache[base] = rp
        return self._robot_cache[base].can_fetch(user_agent, url)


def _extract_links(text: str, base_url: str) -> List[str]:
    links: List[str] = []
    for token in text.split():
        if "href=" not in token:
            continue
        start = token.find("href=")
        if start == -1:
            continue
        value = token[start + 5 :].strip("\"' >")
        if value.startswith("#"):
            continue
        absolute = parse.urljoin(base_url, value)
        if absolute.startswith("http"):
            links.append(absolute)
    return links
