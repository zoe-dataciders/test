import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .paper_search import search_papers
from .pdf_analyzer import analyze_paper


def load_secret_from_key_vault(vault_url: str, secret_name: str) -> str:
    """
    Load one secret value from Azure Key Vault using managed identity/default Azure credential.
    Returns an empty string if loading fails.
    """
    if not vault_url or not secret_name:
        return ""

    try:
        from azure.identity import DefaultAzureCredential  # noqa: PLC0415
        from azure.keyvault.secrets import SecretClient  # noqa: PLC0415
    except ImportError:
        print(
            "Warning: azure-identity/azure-keyvault-secrets not installed, skipping Key Vault lookup.",
            file=sys.stderr,
        )
        return ""

    try:
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=vault_url, credential=credential)
        return client.get_secret(secret_name).value or ""
    except Exception as exc:
        print(f"Warning: failed to read secret '{secret_name}' from Key Vault: {exc}", file=sys.stderr)
        return ""


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
    # Azure OpenAI — all three are required together to enable PDF analysis.
    # If omitted, the program runs without PDF analysis (fast mode).
    # Values can also be supplied via environment variables instead of CLI args.
    parser.add_argument(
        "--analyze-pdfs",
        action="store_true",
        help="Download and analyse PDFs with Azure OpenAI to extract keywords, "
             "results summary, and methods summary.",
    )
    parser.add_argument(
        "--azure-openai-endpoint",
        default=None,
        help="Azure OpenAI endpoint URL, e.g. https://<resource>.openai.azure.com/. "
             "Can also be set via AZURE_OPENAI_ENDPOINT environment variable.",
    )
    parser.add_argument(
        "--azure-openai-key",
        default=None,
        help="Azure OpenAI API key. Can also be set via AZURE_OPENAI_KEY environment variable.",
    )
    parser.add_argument(
        "--azure-openai-deployment",
        default=None,
        help="Azure OpenAI model deployment name (e.g. gpt-4o). "
             "Can also be set via AZURE_OPENAI_DEPLOYMENT environment variable.",
    )
    parser.add_argument(
        "--azure-openai-api-version",
        default="2024-02-01",
        help="Azure OpenAI API version string (default: 2024-02-01).",
    )
    parser.add_argument(
        "--azure-keyvault-url",
        default="https://kv-training-zfl-dev.vault.azure.net/",
        help="Azure Key Vault URL (default: kv-training-zfl-dev). "
             "Can also be set via AZURE_KEYVAULT_URL.",
    )
    parser.add_argument(
        "--azure-openai-key-secret-name",
        default="openai-api-key",
        help="Key Vault secret name for Azure OpenAI API key (default: openai-api-key). "
             "Can also be set via AZURE_OPENAI_KEY_SECRET_NAME.",
    )
    parser.add_argument(
        "--azure-openai-endpoint-secret-name",
        default="azure-openai-endpoint",
        help="Key Vault secret name for Azure OpenAI endpoint (default: azure-openai-endpoint). "
             "Can also be set via AZURE_OPENAI_ENDPOINT_SECRET_NAME.",
    )
    parser.add_argument(
        "--azure-openai-deployment-secret-name",
        default="azure-openai-deployment",
        help="Key Vault secret name for Azure OpenAI deployment (default: azure-openai-deployment). "
             "Can also be set via AZURE_OPENAI_DEPLOYMENT_SECRET_NAME.",
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

    output_dir = Path(args.output_dir).resolve()
    if str(output_dir).startswith("/mnt/"):
        fallback_dir = Path("/tmp/outputs")
        print(
            f"Warning: output directory '{output_dir}' is mounted under /mnt and may fail for delta writes; "
            f"redirecting to '{fallback_dir}'.",
            file=sys.stderr,
        )
        output_dir = fallback_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse topics from comma-separated string
    topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    if not topics:
        print("Error: No topics provided.", file=sys.stderr)
        return 1

    # Build Azure OpenAI client if PDF analysis is requested
    azure_client = None
    azure_deployment = None
    if args.analyze_pdfs:
        endpoint = args.azure_openai_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        api_key = args.azure_openai_key or os.environ.get("AZURE_OPENAI_KEY", "")
        azure_deployment = args.azure_openai_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")

        vault_url = args.azure_keyvault_url or os.environ.get("AZURE_KEYVAULT_URL", "")
        key_secret_name = os.environ.get("AZURE_OPENAI_KEY_SECRET_NAME", args.azure_openai_key_secret_name)
        endpoint_secret_name = os.environ.get(
            "AZURE_OPENAI_ENDPOINT_SECRET_NAME", args.azure_openai_endpoint_secret_name
        )
        deployment_secret_name = os.environ.get(
            "AZURE_OPENAI_DEPLOYMENT_SECRET_NAME", args.azure_openai_deployment_secret_name
        )

        # If values are missing, try to fetch them from Key Vault.
        if vault_url:
            if not endpoint:
                endpoint = load_secret_from_key_vault(vault_url, endpoint_secret_name)
            if not api_key:
                api_key = load_secret_from_key_vault(vault_url, key_secret_name)
            if not azure_deployment:
                azure_deployment = load_secret_from_key_vault(vault_url, deployment_secret_name)

        if not endpoint or not api_key or not azure_deployment:
            print(
                "Error: --analyze-pdfs requires endpoint, key, and deployment values. "
                "Provide them via CLI args, environment variables, or Key Vault.",
                file=sys.stderr,
            )
            return 1

        try:
            from openai import AzureOpenAI  # noqa: PLC0415
            azure_client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=args.azure_openai_api_version,
            )
        except ImportError:
            print("Error: openai package not installed. Run: pip install openai", file=sys.stderr)
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

        # PDF analysis — enrich each row with keywords, results summary, methods summary
        if azure_client is not None:
            import requests as _requests  # noqa: PLC0415

            session = _requests.Session()
            session.headers.update({"User-Agent": "paper-topic-search/1.0 (Python requests)"})

            keywords_col: list[str] = []
            results_col: list[str] = []
            methods_col: list[str] = []

            total = len(results)
            for i, row in enumerate(results.itertuples(), start=1):
                pdf_url = getattr(row, "pdf_url", "") or ""
                title = getattr(row, "title", "") or ""
                print(f"  Analysing PDF {i}/{total}: {title[:60]}...")
                analysis = analyze_paper(
                    title=title,
                    pdf_url=pdf_url,
                    session=session,
                    client=azure_client,
                    deployment=azure_deployment,
                )
                keywords_col.append(analysis["keywords"])
                results_col.append(analysis["results_summary"])
                methods_col.append(analysis["methods_summary"])

            results = results.copy()
            results["pdf_keywords"] = keywords_col
            results["pdf_results_summary"] = results_col
            results["pdf_methods_summary"] = methods_col

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
