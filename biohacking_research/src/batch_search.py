import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .paper_search import search_papers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batch paper searches and write results to specified directory.")
    parser.add_argument("--topics", required=True, help="Comma-separated list of topics to search, e.g., 'gene therapy,CRISPR,longevity'")
    parser.add_argument("--from-date", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--to-date", required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument("--max-results-per-source", type=int, default=100)
    parser.add_argument("--bm25-weight", type=float, default=0.45)
    parser.add_argument("--semantic-weight", type=float, default=0.55)
    parser.add_argument("--use-cross-encoder", action="store_true")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save results to")
    parser.add_argument(
        "--output-format",
        choices=("csv", "parquet", "delta", "all"),
        default="all",
        help="Storage format to write. Use delta for a lakehouse table folder.",
    )
    return parser


def write_results(results, output_dir: Path, output_format: str, table_name: str) -> list[str]:
    written_paths: list[str] = []

    if output_format in {"csv", "all"}:
        csv_path = output_dir / f"{table_name}.csv"
        results.to_csv(csv_path, index=False)
        written_paths.append(str(csv_path))

    if output_format in {"parquet", "all"}:
        parquet_path = output_dir / f"{table_name}.parquet"
        results.to_parquet(parquet_path, index=False)
        written_paths.append(str(parquet_path))

    if output_format in {"delta", "all"}:
        try:
            import pyarrow as pa
            from deltalake.writer import write_deltalake

            delta_path = output_dir / table_name
            delta_table = pa.Table.from_pandas(results, preserve_index=False)
            write_deltalake(str(delta_path), delta_table, mode="overwrite")
            written_paths.append(str(delta_path))
        except ImportError:
            print("Warning: pyarrow or deltalake not installed, skipping delta format.", file=sys.stderr)

    return written_paths


def main() -> int:
    args = build_parser().parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse topics from comma-separated string
    topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    if not topics:
        print("Error: No topics provided.", file=sys.stderr)
        return 1

    all_metadata = []

    for topic in topics:
        print(f"Searching for topic: '{topic}'")
        try:
            results = search_papers(
                topic=topic,
                from_date=args.from_date,
                to_date=args.to_date,
                max_results_per_source=args.max_results_per_source,
                bm25_weight=args.bm25_weight,
                semantic_weight=args.semantic_weight,
                use_cross_encoder=args.use_cross_encoder,
            )
        except Exception as exc:
            print(f"Error searching for '{topic}': {exc}", file=sys.stderr)
            continue

        if results.empty:
            print(f"  No results found for '{topic}'")
            all_metadata.append({
                "topic": topic,
                "row_count": 0,
                "status": "no results",
            })
            continue

        # Use topic name as table name (replace spaces with underscores)
        table_name = topic.replace(" ", "_").lower()
        
        written_paths = write_results(
            results=results,
            output_dir=output_dir,
            output_format=args.output_format,
            table_name=table_name,
        )

        print(f"  Wrote {len(results)} rows for '{topic}':")
        for path in written_paths:
            print(f"    - {path}")

        all_metadata.append({
            "topic": topic,
            "from_date": args.from_date,
            "to_date": args.to_date,
            "max_results_per_source": args.max_results_per_source,
            "bm25_weight": args.bm25_weight,
            "semantic_weight": args.semantic_weight,
            "use_cross_encoder": args.use_cross_encoder,
            "output_format": args.output_format,
            "table_name": table_name,
            "row_count": len(results),
            "status": "success",
            "written_paths": written_paths,
        })

    # Write summary metadata
    metadata = {
        "topics_searched": topics,
        "total_topics": len(topics),
        "results": all_metadata,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path = output_dir / "batch_search_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSummary written to {metadata_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
