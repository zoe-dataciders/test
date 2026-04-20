"""
PDF Analyzer — download a research paper PDF, extract its text,
and use Azure OpenAI to extract structured information:
  - keywords
  - summary of results
  - summary of methods
"""

import io
import json
import sys

import requests


# Maximum characters of PDF text sent to the LLM.
# A typical research paper is ~50 000 chars; we cap at 40 000 to stay well
# within the GPT-4 context window while covering the full paper body.
MAX_TEXT_CHARS = 40_000

# PDF pages to read (most methods + results are in the first 30 pages).
MAX_PDF_PAGES = 30


def download_pdf(pdf_url: str, session: requests.Session, timeout: int = 30) -> bytes | None:
    """
    Download the PDF at *pdf_url* and return the raw bytes.
    Returns None if the download fails for any reason.
    """
    try:
        response = session.get(pdf_url, timeout=timeout)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "pdf" not in content_type and not pdf_url.endswith(".pdf"):
            # Guard against accidentally fetching an HTML error page
            return None
        return response.content
    except Exception:
        return None


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract plain text from a PDF given as raw bytes.
    Uses *pdfplumber*; returns an empty string on any error.
    """
    try:
        import pdfplumber  # noqa: PLC0415 – imported lazily to keep startup fast
    except ImportError:
        print("Warning: pdfplumber not installed. Run: pip install pdfplumber", file=sys.stderr)
        return ""

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages_text: list[str] = []
            for page in pdf.pages[:MAX_PDF_PAGES]:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            return "\n".join(pages_text)
    except Exception:
        return ""


def analyze_with_azure_openai(
    text: str,
    title: str,
    client,  # openai.AzureOpenAI
    deployment: str,
) -> dict:
    """
    Send the paper text to Azure OpenAI and extract structured fields.

    Returns a dict with keys: keywords, results_summary, methods_summary.
    All values are plain strings; empty strings are returned on any error.
    """
    EMPTY = {"keywords": "", "results_summary": "", "methods_summary": ""}

    # Truncate text to stay inside the model context window
    truncated = text[:MAX_TEXT_CHARS]

    prompt = (
        "You are a biomedical research analyst. Analyse the following research paper "
        "and extract exactly the three items below.\n\n"
        "1. keywords – a comma-separated list of the most important medical/scientific "
        "terms, interventions, biomarkers, and compounds mentioned in the paper.\n"
        "2. results_summary – 2-3 sentences describing the main findings and outcomes.\n"
        "3. methods_summary – 2-3 sentences describing the study design, population, "
        "and experimental or analytical methods used.\n\n"
        f"Paper title: {title}\n\n"
        f"Paper content:\n{truncated}\n\n"
        "Respond with ONLY a JSON object in this exact format:\n"
        '{"keywords": "...", "results_summary": "...", "methods_summary": "..."}'
    )

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        parsed = json.loads(raw)
        return {
            "keywords": str(parsed.get("keywords", "")),
            "results_summary": str(parsed.get("results_summary", "")),
            "methods_summary": str(parsed.get("methods_summary", "")),
        }
    except Exception as exc:
        print(f"  Warning: Azure OpenAI analysis failed — {exc}", file=sys.stderr)
        return EMPTY


def analyze_paper(
    title: str,
    pdf_url: str,
    session: requests.Session,
    client,  # openai.AzureOpenAI
    deployment: str,
) -> dict:
    """
    Full pipeline for one paper:
      1. Download PDF
      2. Extract text
      3. Analyse with Azure OpenAI
    Returns a dict with keywords, results_summary, methods_summary.
    Empty strings are used for any step that fails.
    """
    EMPTY = {"keywords": "", "results_summary": "", "methods_summary": ""}

    if not pdf_url:
        return EMPTY

    pdf_bytes = download_pdf(pdf_url, session)
    if not pdf_bytes:
        print(f"  Warning: could not download PDF for '{title}'", file=sys.stderr)
        return EMPTY

    text = extract_text_from_pdf(pdf_bytes)
    if not text:
        print(f"  Warning: could not extract text from PDF for '{title}'", file=sys.stderr)
        return EMPTY

    return analyze_with_azure_openai(text, title, client, deployment)
