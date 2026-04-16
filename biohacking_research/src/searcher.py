import time
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

import pandas as pd
import requests

from .models import PaperResult
from .ranker import HybridRanker
from .utils import build_arxiv_query, deduplicate_results, normalize_space, parse_datetime, parse_timeframe


class PaperSearcher:
    """
    This class is responsible for collecting papers from the APIs.
    """

    def __init__( # we pass all the arguments that will be inserted into self.x when the class is instantiated.
        self, # self gets replaced by the name of the instance once the class is instantiated
        timeout: int = 10,
        pause_seconds: float = 0.2,
        candidate_multiplier: int = 5,
        max_api_pages_per_source: int = 8,
        ranker: HybridRanker | None = None,
    ) -> None:
        

        """
        self will be replaced by object name once the class is instantiated and then will
        automatically be passed the above defined values from the arguments.
        """
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({"User-Agent": "paper-topic-search/1.0 (Python requests)"})

        self.timeout = timeout
        self.pause_seconds = pause_seconds
        self.candidate_multiplier = max(candidate_multiplier, 2)
        self.max_api_pages_per_source = max_api_pages_per_source
        self.ranker = ranker or HybridRanker()

    def search(self, topic: str, from_date: str, to_date: str, max_results_per_source: int = 100) -> pd.DataFrame:
        """
        Main search method.
        It fetches candidate papers, removes duplicates, ranks them,
        and returns a DataFrame.
        """

        start_date = datetime.fromisoformat(from_date)
        end_date = datetime.fromisoformat(to_date)
        all_results: list[PaperResult] = []

        source_searchers = [
            self.search_biorxiv,
            self.search_medrxiv,
            self.search_arxiv,
        ]

        for source_searcher in source_searchers:
            try:
                all_results.extend(source_searcher(topic, start_date, end_date, max_results_per_source))
            except (requests.RequestException, ET.ParseError, ValueError):
                continue

        deduped = deduplicate_results(all_results)
        ranked = self.ranker.rerank(topic, deduped)

        rows = [
            {
                "name of research paper": f"{paper.title} ({paper.link})",
                "date published": paper.published.date().isoformat(),
                "relevance": round(paper.relevance, 3),
            }
            for paper in ranked
        ]

        return pd.DataFrame(rows, columns=["name of research paper", "date published", "relevance"])

    def search_biorxiv(self, topic: str, start_date: datetime, end_date: datetime, max_results: int) -> list[PaperResult]:
        """
        Search bioRxiv.
        """

        return self._search_biorxiv_family("biorxiv", start_date, end_date, max_results)

    def search_medrxiv(self, topic: str, start_date: datetime, end_date: datetime, max_results: int) -> list[PaperResult]:
        """
        Search medRxiv.
        """

        return self._search_biorxiv_family("medrxiv", start_date, end_date, max_results)

    def _search_biorxiv_family(
        self,
        server: str,
        start_date: datetime,
        end_date: datetime,
        max_results: int,
    ) -> list[PaperResult]:
        """
        Shared helper for bioRxiv and medRxiv.
        bioRxiv and medRxiv use almost the same API,
        so instead of writing the same code twice,
        the program puts the shared logic in one function and
        passes in which server to use.
        """

        end_date_str = end_date.date().isoformat()
        start_date_str = start_date.date().isoformat()
        cursor = 0
        page_count = 0
        candidate_limit = max_results * self.candidate_multiplier
        results: list[PaperResult] = []

        while len(results) < candidate_limit and page_count < self.max_api_pages_per_source:
            url = f"https://api.biorxiv.org/details/{server}/{start_date_str}/{end_date}/{cursor}"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()

            collection = payload.get("collection", [])
            if not collection:
                break

            for item in collection:
                published = parse_datetime(item.get("date"))
                if not published or published < start_date:
                    continue

                doi = item.get("doi", "")
                if doi:
                    if doi.startswith("10.1101/"):
                        link = f"https://doi.org/{doi}"
                    else:
                        link = f"https://doi.org/10.1101/{doi}"
                else:
                    link = item.get("url", "")

                results.append(
                    PaperResult(
                        title=normalize_space(item.get("title", "")),
                        published=published,
                        relevance=0.0,
                        link=link,
                        source=server,
                        abstract=normalize_space(item.get("abstract", "")),
                    )
                )

                if len(results) >= candidate_limit:
                    break

            messages = payload.get("messages", [])
            if not messages:
                break

            try:
                total_items = int(messages[0].get("total", "0"))
            except (TypeError, ValueError):
                total_items = 0

            cursor += len(collection)
            page_count += 1

            if cursor >= total_items:
                break

            time.sleep(self.pause_seconds)

        return results

    def search_arxiv(self, topic: str, start_date: datetime, end_date: datetime, max_results: int) -> list[PaperResult]:
        """
        Search arXiv.

        arXiv returns XML instead of JSON, so the parsing looks different.
        """

        url = "https://export.arxiv.org/api/query"
        page_size = 100
        start_index = 0
        page_count = 0
        candidate_limit = max_results * self.candidate_multiplier
        results: list[PaperResult] = []
        namespace = {"atom": "http://www.w3.org/2005/Atom"}

        while len(results) < candidate_limit and page_count < self.max_api_pages_per_source:
            params = {
                "search_query": build_arxiv_query(topic),
                "start": start_index,
                "max_results": page_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "submittedDate": f"[{start_date.date()} TO {end_date.date()}]",
            }

            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            root = ET.fromstring(response.text)

            entries = root.findall("atom:entry", namespace)
            if not entries:
                break

            saw_in_range_entry = False
            for entry in entries:
                published_text = entry.findtext("atom:published", default="", namespaces=namespace)
                published = parse_datetime(published_text)
                if not published:
                    continue
                if published < start_date:
                    continue

                saw_in_range_entry = True

                title = normalize_space(entry.findtext("atom:title", default="", namespaces=namespace))
                abstract = normalize_space(entry.findtext("atom:summary", default="", namespaces=namespace))

                link = ""
                for link_el in entry.findall("atom:link", namespace):
                    if link_el.attrib.get("rel") == "alternate":
                        link = link_el.attrib.get("href", "")
                        break

                if not link:
                    link = entry.findtext("atom:id", default="", namespaces=namespace)

                results.append(
                    PaperResult(
                        title=title,
                        published=published,
                        relevance=0.0,
                        link=link,
                        source="arxiv",
                        abstract=abstract,
                    )
                )

                if len(results) >= candidate_limit:
                    break

            if not saw_in_range_entry:
                break

            start_index += page_size
            page_count += 1
            time.sleep(self.pause_seconds)

        return results