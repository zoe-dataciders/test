from dataclasses import dataclass
from datetime import datetime


@dataclass
class PaperResult:
    """
    One PaperResult object represents one research paper.

    Because this uses @dataclass, Python automatically creates common setup code
    for us, especially the __init__ method.

    __init__ is the method that runs when an object is created.
    """

    title: str
    published: datetime
    relevance: float
    link: str
    source: str
    abstract: str = ""
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    cross_encoder_score: float = 0.0


@dataclass
class RankedDocument:
    """
    This is a helper object for BM25 ranking.

    It stores the paper plus tokenized versions of the title and abstract.
    "Tokenized" means text has been split into simplified word-like pieces.
    """

    paper: PaperResult
    title_tokens: list[str]
    abstract_tokens: list[str]
    combined_tokens: list[str]