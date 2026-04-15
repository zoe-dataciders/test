import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable

from .models import PaperResult, RankedDocument


def parse_timeframe(timeframe: str) -> datetime:
    """
    Turn human-friendly timeframe text into a datetime cutoff.
    """

    text = timeframe.strip().lower()
    match = re.search(r"(?P<count>\d+)\s*(?P<unit>day|days|week|weeks|month|months|year|years)", text)

    if not match:
        raise ValueError("Timeframe must look like 'last 5 years', '2 months', or '10 days'.")

    count = int(match.group("count"))
    unit = match.group("unit")
    now = datetime.now(timezone.utc)

    if "day" in unit:
        return now - timedelta(days=count)
    if "week" in unit:
        return now - timedelta(weeks=count)
    if "month" in unit:
        return now - timedelta(days=count * 30)
    return now - timedelta(days=count * 365)


def parse_datetime(value: str) -> datetime | None:
    """
    Try to convert a string date into a real datetime object.
    """

    if not value:
        return None

    raw = value.strip()

    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%b %d, %Y", "%B %d, %Y"):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue

    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def score_bm25(topic: str, results: list[PaperResult]) -> list[PaperResult]:
    """
    Compute BM25-style lexical scores.

    This is the first ranking stage.
    It focuses on words and word overlap.
    """

    query_terms = tokenize(topic)
    if not query_terms:
        return sorted(results, key=lambda item: (-item.published.timestamp(), item.title.lower()))

    documents = [
        RankedDocument(
            paper=paper,
            title_tokens=tokenize(paper.title),
            abstract_tokens=tokenize(paper.abstract),
            combined_tokens=tokenize(f"{paper.title} {paper.abstract}"),
        )
        for paper in results
    ]

    documents = [doc for doc in documents if doc.combined_tokens]
    if not documents:
        return []

    avg_doc_len = sum(len(doc.combined_tokens) for doc in documents) / len(documents)

    # Document frequency means:
    # in how many different documents does a query term appear?
    doc_freq = Counter()
    for doc in documents:
        for term in set(query_terms):
            if term in doc.combined_tokens:
                doc_freq[term] += 1

    newest = max(doc.paper.published for doc in documents)
    oldest = min(doc.paper.published for doc in documents)
    span_seconds = max((newest - oldest).total_seconds(), 1.0)
    normalized_query = normalize_space(topic.lower())

    ranked: list[PaperResult] = []
    for doc in documents:
        title_counter = Counter(doc.title_tokens)
        abstract_counter = Counter(doc.abstract_tokens)
        combined_counter = Counter(doc.combined_tokens)

        bm25_score = 0.0
        coverage = 0

        for term in query_terms:
            tf = combined_counter[term]
            if tf == 0:
                continue

            coverage += 1
            df = doc_freq[term]
            idf = math.log(1 + ((len(documents) - df + 0.5) / (df + 0.5)))
            numerator = tf * (1.5 + 1)
            denominator = tf + 1.5 * (1 - 0.75 + 0.75 * (len(doc.combined_tokens) / avg_doc_len))
            bm25_score += idf * (numerator / denominator)

        # These are extra practical bonuses layered on top of BM25.
        title_hits = sum(title_counter[term] for term in query_terms)
        abstract_hits = sum(abstract_counter[term] for term in query_terms)
        title_bonus = 1.8 * title_hits
        abstract_bonus = 0.35 * abstract_hits
        phrase_bonus = 4.0 if normalized_query in normalize_space(f"{doc.paper.title} {doc.paper.abstract}".lower()) else 0.0
        title_phrase_bonus = 2.5 if normalized_query in normalize_space(doc.paper.title.lower()) else 0.0
        coverage_bonus = (coverage / len(set(query_terms))) * 3.0
        recency_bonus = ((doc.paper.published - oldest).total_seconds() / span_seconds) * 0.5

        lexical_score = (
            bm25_score + title_bonus + abstract_bonus + phrase_bonus + title_phrase_bonus + coverage_bonus + recency_bonus
        )

        doc.paper.bm25_score = round(lexical_score, 4)
        doc.paper.relevance = doc.paper.bm25_score
        ranked.append(doc.paper)

    return sorted(ranked, key=lambda item: (-item.bm25_score, -item.published.timestamp(), item.title.lower()))


def deduplicate_results(results: Iterable[PaperResult]) -> list[PaperResult]:
    """
    Remove duplicate papers based on normalized title text.
    """

    best_by_key: dict[str, PaperResult] = {}
    for item in results:
        key = normalize_space(item.title).lower()
        existing = best_by_key.get(key)

        if existing is None or (item.relevance, item.published) > (existing.relevance, existing.published):
            best_by_key[key] = item

    return list(best_by_key.values())


def normalize_space(value: str) -> str:
    """
    Replace repeated whitespace with single spaces and trim the ends.
    """

    return re.sub(r"\s+", " ", value or "").strip()


def normalize_cosine_similarity(value: float) -> float:
    """
    Convert cosine similarity from [-1, 1] into [0, 1].
    """

    return max(0.0, min(1.0, (value + 1.0) / 2.0))


def normalize_score_map(values: list[float]) -> list[float]:
    """
    Min-max normalize a list of numeric scores into a 0-to-1 range.
    """

    if not values:
        return []

    max_value = max(values)
    min_value = min(values)
    span = max(max_value - min_value, 1e-9)
    return [(value - min_value) / span for value in values]


def build_semantic_text(paper: PaperResult) -> str:
    """
    Create the text that will be converted into an embedding.
    """

    parts = [paper.title, paper.abstract, paper.source]
    return normalize_space(". ".join(part for part in parts if part))


def build_arxiv_query(topic: str) -> str:
    """
    Build an arXiv query string.
    """

    query_terms = [term for term in tokenize(topic) if len(term) > 2]
    if not query_terms:
        return f'all:"{topic}"'

    expanded_terms = [topic] + query_terms
    quoted_terms = [f'all:"{term}"' for term in dict.fromkeys(expanded_terms)]
    return " OR ".join(quoted_terms)


def tokenize(value: str) -> list[str]:
    """
    Break text into simplified word pieces and stem them.
    """

    return [stem_token(token) for token in re.findall(r"[a-z0-9]+", (value or "").lower()) if token]


def stem_token(token: str) -> str:
    """
    Simplify a word to a rough stem so related word forms match better.
    """

    stemmed = token

    for suffix in ("ization", "ational", "fulness", "ousness", "iveness", "tional"):
        if stemmed.endswith(suffix) and len(stemmed) > len(suffix) + 3:
            stemmed = stemmed[: -len(suffix)]
            break

    for suffix in ("ologist", "ological", "ologic", "ology", "ically", "ality"):
        if stemmed.endswith(suffix) and len(stemmed) > len(suffix) + 3:
            stemmed = stemmed[: -len(suffix)]
            break

    for suffix in ("ingly", "edly", "ments", "ment", "ness", "less", "able", "ible", "tion", "sion"):
        if stemmed.endswith(suffix) and len(stemmed) > len(suffix) + 3:
            stemmed = stemmed[: -len(suffix)]
            break

    for suffix in ("ical", "ally", "ably", "ibly", "ics", "ist", "ism", "ing", "ied", "ies", "ed", "ic", "al", "ly", "es", "s", "y"):
        if stemmed.endswith(suffix) and len(stemmed) > len(suffix) + 2:
            stemmed = stemmed[: -len(suffix)]
            break

    return stemmed