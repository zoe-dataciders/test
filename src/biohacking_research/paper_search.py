import argparse
import sys

import pandas as pd

from .ranker import DEFAULT_CROSS_ENCODER_MODEL, DEFAULT_EMBEDDING_MODEL, HybridRanker
from .searcher import PaperSearcher


def search_papers(
    topic: str,
    from_date: str,
    to_date: str,
    max_results_per_source: int = 100,
    bm25_weight: float = 0.45,
    semantic_weight: float = 0.55,
    use_cross_encoder: bool = False,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    cross_encoder_model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
) -> pd.DataFrame:
    """
    Main function for other Python code to call.
    """

    ranker = HybridRanker(
        embedding_model_name=embedding_model_name,
        bm25_weight=bm25_weight,
        semantic_weight=semantic_weight,
        use_cross_encoder=use_cross_encoder,
        cross_encoder_model_name=cross_encoder_model_name,
    )

    return PaperSearcher(ranker=ranker).search(
        topic=topic,
        from_date=from_date,
        to_date=to_date,
        max_results_per_source=max_results_per_source,
    )


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser.

    This tells Python what inputs are allowed when the script is run
    from the terminal.
    """

    parser = argparse.ArgumentParser(
        description="Search bioRxiv, medRxiv, and arXiv for papers on a topic."
    )

    parser.add_argument("topic", help="Topic to search for, for example 'neurology'")
    parser.add_argument("--from-date", required=True, help="Start date in YYYY-MM-DD format, e.g., 2020-01-01")
    parser.add_argument("--to-date", required=True, help="End date in YYYY-MM-DD format, e.g., 2024-12-31")

    parser.add_argument(
        "--max-results-per-source",
        type=int,
        default=100,
        help="Maximum number of matching papers to keep from each source.",
    )

    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=0.45,
        help="Weight for lexical BM25 relevance in the hybrid score.",
    )

    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.55,
        help="Weight for embedding similarity in the hybrid score.",
    )

    parser.add_argument(
        "--use-cross-encoder",
        action="store_true",
        help="Use an optional cross-encoder reranker on the top semantic results.",
    )

    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model name for semantic embeddings.",
    )

    parser.add_argument(
        "--cross-encoder-model",
        default=DEFAULT_CROSS_ENCODER_MODEL,
        help="Cross-encoder model name for the optional final reranking step.",
    )

    return parser


def main() -> int:
    """
    Standard script entry point.

    It reads the command-line arguments, runs the search, and prints the result.
    """

    parser = build_parser()
    args = parser.parse_args()

    try:
        df = search_papers(
            topic=args.topic,
            from_date=args.from_date,
            to_date=args.to_date,
            max_results_per_source=args.max_results_per_source,
            bm25_weight=args.bm25_weight,
            semantic_weight=args.semantic_weight,
            use_cross_encoder=args.use_cross_encoder,
            embedding_model_name=args.embedding_model,
            cross_encoder_model_name=args.cross_encoder_model,
        )
    except Exception as exc:
        print(f"Search failed: {exc}", file=sys.stderr)
        return 1

    if df.empty:
        print("No matching papers found.")
    else:
        print(df.to_string(index=False))

    return 0


# __name__ is a special Python variable.
# If this file is run directly, __name__ becomes "__main__".
# If the file is imported into another file, __name__ is something else.
# This line makes sure main() only runs when this file is executed directly.
if __name__ == "__main__":
    raise SystemExit(main())