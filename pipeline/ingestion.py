"""
pipeline/ingestion.py
Document ingestion: fetch → parse → clean → save with rich metadata.

Retrieval accuracy depends heavily on metadata quality — source authority,
recency, and specialty allow filtered retrieval that boosts precision.
"""

import json
import hashlib
import requests
import fitz  # PyMuPDF
from pathlib import Path
from datetime import datetime
from typing import Optional
from bs4 import BeautifulSoup
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import RAW_DIR, PROCESSED_DIR, TRUSTED_SOURCES


# ─── Data model ───────────────────────────────────────────────────────────────

class MedicalDocument:
    """
    A parsed medical document with rich metadata for retrieval filtering.

    Metadata fields directly impact retrieval quality:
    - source_authority: enables filtering to high-credibility sources only
    - specialty: allows domain-scoped retrieval (e.g., only cardiology docs)
    - evidence_level: lets the prompt tier citations by strength
    - pub_date: supports recency-weighted retrieval
    """

    def __init__(
        self,
        content: str,
        title: str,
        source_url: str,
        source_name: str,
        source_authority: str,       # "primary" | "secondary" | "review"
        specialty: str,              # e.g. "cardiology", "general", "oncology"
        evidence_level: str,         # "guideline" | "rct" | "review" | "factsheet"
        pub_date: Optional[str] = None,
        doc_type: str = "article",   # "article" | "guideline" | "factsheet" | "abstract"
        language: str = "en",
    ):
        self.content = content
        self.title = title
        self.source_url = source_url
        self.source_name = source_name
        self.source_authority = source_authority
        self.specialty = specialty
        self.evidence_level = evidence_level
        self.pub_date = pub_date or datetime.utcnow().strftime("%Y-%m-%d")
        self.doc_type = doc_type
        self.language = language
        self.ingested_at = datetime.utcnow().isoformat()
        # Stable ID for deduplication
        self.doc_id = hashlib.sha256(source_url.encode()).hexdigest()[:16]
        self.word_count = len(content.split())

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "source_url": self.source_url,
            "source_name": self.source_name,
            "source_authority": self.source_authority,
            "specialty": self.specialty,
            "evidence_level": self.evidence_level,
            "pub_date": self.pub_date,
            "doc_type": self.doc_type,
            "language": self.language,
            "ingested_at": self.ingested_at,
            "word_count": len(self.content.split()),
        }


# ─── Parsers ──────────────────────────────────────────────────────────────────

class HTMLParser:
    """Parse HTML pages from trusted medical websites."""

    def parse(self, html: str, url: str) -> str:
        soup = BeautifulSoup(html, "lxml")

        # Remove boilerplate elements that hurt retrieval signal
        for tag in soup(["script", "style", "nav", "footer",
                          "header", "aside", "form", "button"]):
            tag.decompose()

        # Prefer main content area if available
        main = (
            soup.find("article") or
            soup.find("main") or
            soup.find(id="main-content") or
            soup.find(class_="content") or
            soup.body
        )
        text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")
        return self._clean(text)

    def _clean(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines()]
        lines = [l for l in lines if len(l) > 20]  # Drop short nav fragments
        return "\n".join(lines)


class PDFParser:
    """Parse PDF documents (guidelines, clinical references)."""

    def parse(self, pdf_bytes: bytes) -> str:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
        return self._clean("\n".join(pages))

    def _clean(self, text: str) -> str:
        # Remove page headers/footers (typically < 5 words on a line)
        lines = text.splitlines()
        cleaned = []
        for line in lines:
            stripped = line.strip()
            word_count = len(stripped.split())
            if word_count >= 4:
                cleaned.append(stripped)
        return "\n".join(cleaned)


# ─── Fetcher ──────────────────────────────────────────────────────────────────

class DocumentFetcher:
    """Fetch documents from trusted URLs with retry and domain validation."""

    HEADERS = {
        "User-Agent": (
            "MedicalRAGResearchBot/1.0 "
            "(educational use; contact: your@email.com)"
        )
    }

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.html_parser = HTMLParser()
        self.pdf_parser = PDFParser()

    def fetch(self, url: str) -> Optional[str]:
        """Fetch and parse content from a URL. Returns clean text or None."""
        if not self._is_trusted(url):
            logger.warning(f"Skipping untrusted domain: {url}")
            return None

        try:
            response = requests.get(url, headers=self.HEADERS, timeout=self.timeout)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "application/pdf" in content_type:
                return self.pdf_parser.parse(response.content)
            else:
                return self.html_parser.parse(response.text, url)

        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def _is_trusted(self, url: str) -> bool:
        return any(domain in url for domain in TRUSTED_SOURCES)


# ─── Ingestion pipeline ────────────────────────────────────────────────────────

class IngestionPipeline:
    """
    Orchestrates the full ingestion flow:
    fetch → parse → validate → save with metadata.
    """

    def __init__(self):
        self.fetcher = DocumentFetcher()
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    def ingest_from_manifest(self, manifest_path: str) -> list[MedicalDocument]:
        """
        Ingest a batch of documents defined in a JSON manifest.
        Manifest format: list of {url, title, source_name, source_authority,
                                   specialty, evidence_level, doc_type}
        """
        with open(manifest_path) as f:
            manifest = json.load(f)

        docs = []
        for i, entry in enumerate(manifest, 1):
            logger.info(f"[{i}/{len(manifest)}] Ingesting: {entry['title']}")
            content = self.fetcher.fetch(entry["url"])
            if not content or len(content.split()) < 100:
                logger.warning(f"  Skipped (too short or failed): {entry['url']}")
                continue

            doc = MedicalDocument(
                content=content,
                title=entry["title"],
                source_url=entry["url"],
                source_name=entry["source_name"],
                source_authority=entry["source_authority"],
                specialty=entry["specialty"],
                evidence_level=entry["evidence_level"],
                doc_type=entry.get("doc_type", "article"),
                pub_date=entry.get("pub_date"),
            )
            self._save(doc)
            docs.append(doc)
            logger.info(f"  Saved doc_id={doc.doc_id} ({doc.word_count} words)")

        logger.info(f"Ingestion complete: {len(docs)}/{len(manifest)} documents saved.")
        return docs

    def ingest_local_pdf(
        self,
        pdf_path: str,
        title: str,
        source_name: str,
        source_authority: str,
        specialty: str,
        evidence_level: str,
    ) -> Optional[MedicalDocument]:
        """Ingest a locally stored PDF (e.g., downloaded guidelines)."""
        pdf_bytes = Path(pdf_path).read_bytes()
        content = PDFParser().parse(pdf_bytes)
        if len(content.split()) < 100:
            logger.warning(f"PDF too short to be useful: {pdf_path}")
            return None

        doc = MedicalDocument(
            content=content,
            title=title,
            source_url=f"local://{pdf_path}",
            source_name=source_name,
            source_authority=source_authority,
            specialty=specialty,
            evidence_level=evidence_level,
            doc_type="guideline",
        )
        self._save(doc)
        return doc

    def _save(self, doc: MedicalDocument):
        out_path = PROCESSED_DIR / f"{doc.doc_id}.json"
        with open(out_path, "w") as f:
            json.dump(doc.to_dict(), f, indent=2)


# ─── Sample manifest builder ──────────────────────────────────────────────────

def create_sample_manifest(output_path: str):
    """
    Creates a sample manifest with high-quality medical sources.
    Add more entries following this pattern to expand the corpus.
    """
    manifest = [
        {
            "url": "https://medlineplus.gov/diabetes.html",
            "title": "Diabetes — MedlinePlus Overview",
            "source_name": "MedlinePlus (NIH)",
            "source_authority": "primary",
            "specialty": "endocrinology",
            "evidence_level": "factsheet",
            "doc_type": "factsheet",
        },
        {
            "url": "https://www.cdc.gov/heartdisease/facts.htm",
            "title": "Heart Disease Facts — CDC",
            "source_name": "CDC",
            "source_authority": "primary",
            "specialty": "cardiology",
            "evidence_level": "factsheet",
            "doc_type": "factsheet",
        },
        {
            "url": "https://www.who.int/news-room/fact-sheets/detail/hypertension",
            "title": "Hypertension Fact Sheet — WHO",
            "source_name": "WHO",
            "source_authority": "primary",
            "specialty": "cardiology",
            "evidence_level": "guideline",
            "doc_type": "factsheet",
        },
        {
            "url": "https://medlineplus.gov/highbloodpressure.html",
            "title": "High Blood Pressure — MedlinePlus",
            "source_name": "MedlinePlus (NIH)",
            "source_authority": "primary",
            "specialty": "cardiology",
            "evidence_level": "factsheet",
            "doc_type": "factsheet",
        },
        {
            "url": "https://medlineplus.gov/asthma.html",
            "title": "Asthma — MedlinePlus Overview",
            "source_name": "MedlinePlus (NIH)",
            "source_authority": "primary",
            "specialty": "pulmonology",
            "evidence_level": "factsheet",
            "doc_type": "factsheet",
        },
        {
            "url": "https://www.cdc.gov/cancer/cervical/basic_info/symptoms.htm",
            "title": "Cervical Cancer Symptoms — CDC",
            "source_name": "CDC",
            "source_authority": "primary",
            "specialty": "oncology",
            "evidence_level": "factsheet",
            "doc_type": "factsheet",
        },
    ]

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Sample manifest written to {output_path} ({len(manifest)} entries)")


if __name__ == "__main__":
    # Example: create manifest and run ingestion
    manifest_path = str(RAW_DIR / "manifest.json")
    create_sample_manifest(manifest_path)
    pipeline = IngestionPipeline()
    docs = pipeline.ingest_from_manifest(manifest_path)
    print(f"\nIngested {len(docs)} documents.")
