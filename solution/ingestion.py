"""
PDF Ingestion Pipeline

Recursively discovers and processes PDF documents in parallel.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from config import LOG_LEVEL
from pdf_parser import parse_single
from chunker import chunk_text_simple
from embeddings import create_embedding
from db import insert_knowledge_base_entry

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def discover_pdfs(directory: str | Path, recursive: bool = True) -> list[Path]:
    """
    Discover all PDF files in a directory.

    Args:
        directory: Path to the directory to search.
        recursive: If True, search subdirectories recursively.

    Returns:
        List of Path objects for each PDF found.
    """
    dir_path = Path(directory).resolve()

    if not dir_path.is_dir():
        raise ValueError(f"Not a valid directory: {dir_path}")

    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = sorted(dir_path.glob(pattern))

    logger.info(f"Discovered {len(pdf_files)} PDF files in {dir_path}")
    return pdf_files


def process_single_pdf(pdf_path: Path, base_dir: Path) -> dict[str, Any]:
    """
    Process a single PDF: parse, chunk, embed, and store.

    Args:
        pdf_path: Path to the PDF file.
        base_dir: Base directory for computing relative path (for source_file).

    Returns:
        Dict with processing results and statistics.
    """
    # Compute relative path for citation
    source_file = str(pdf_path.relative_to(base_dir))

    result = {
        "file_path": str(pdf_path),
        "source_file": source_file,
        "success": False,
        "chunks_created": 0,
        "entries_inserted": 0,
        "error": None,
    }

    try:
        # Parse PDF
        parse_result = parse_single(pdf_path)

        if parse_result.get("error"):
            result["error"] = parse_result["error"]
            return result

        # Extract and clean text
        content = parse_result["text"].replace("\n", "  ")

        if not content.strip():
            result["error"] = "No text content extracted"
            return result

        # Chunk the text
        chunks = chunk_text_simple(content)
        result["chunks_created"] = len(chunks)

        # Create embeddings and insert into database
        for chunk in chunks:
            embedding = create_embedding(chunk)
            insert_knowledge_base_entry(
                text=chunk,
                embedding=embedding,
                num_words=len(chunk.split()),
                text_length=len(chunk),
                source_file=source_file,
            )
            result["entries_inserted"] += 1

        result["success"] = True
        logger.info(f"Processed {source_file}: {result['entries_inserted']} entries")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error processing {source_file}: {e}")

    return result


def ingest_directory(
    directory: str | Path,
    *,
    recursive: bool = True,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """
    Ingest all PDFs from a directory in parallel.

    Args:
        directory: Path to the directory containing PDFs.
        recursive: If True, search subdirectories recursively.
        max_workers: Maximum number of parallel workers.
                    Defaults to min(32, cpu_count + 4).

    Returns:
        Dict with overall ingestion statistics:
            - total_files: Number of PDFs found
            - successful: Number successfully processed
            - failed: Number that failed
            - total_chunks: Total chunks created
            - total_entries: Total database entries inserted
            - results: List of per-file results
            - errors: List of files that had errors
    """
    # Resolve base directory
    base_dir = Path(directory).resolve()

    # Discover PDFs
    pdf_files = discover_pdfs(directory, recursive=recursive)

    if not pdf_files:
        logger.warning(f"No PDF files found in {directory}")
        return {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "total_chunks": 0,
            "total_entries": 0,
            "results": [],
            "errors": [],
        }

    # Process PDFs in parallel
    logger.info(f"Starting parallel ingestion of {len(pdf_files)} PDFs...")

    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(process_single_pdf, pdf_path, base_dir): pdf_path
            for pdf_path in pdf_files
        }

        # Collect results as they complete
        for future in as_completed(future_to_path):
            pdf_path = future_to_path[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(
                    {
                        "file_path": str(pdf_path),
                        "success": False,
                        "chunks_created": 0,
                        "entries_inserted": 0,
                        "error": str(e),
                    }
                )
                logger.error(f"Unexpected error processing {pdf_path.name}: {e}")

    # Compute summary statistics
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    summary = {
        "total_files": len(pdf_files),
        "successful": len(successful),
        "failed": len(failed),
        "total_chunks": sum(r["chunks_created"] for r in results),
        "total_entries": sum(r["entries_inserted"] for r in results),
        "results": results,
        "errors": [{"file": r["file_path"], "error": r["error"]} for r in failed],
    }

    logger.info(
        f"Ingestion complete: {summary['successful']}/{summary['total_files']} successful, "
        f"{summary['total_entries']} entries inserted"
    )

    return summary


def main():
    """Main entry point for the ingestion pipeline."""

    summary = ingest_directory(
        "../novatech-kb",
        recursive=True,
        max_workers=None,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("INGESTION SUMMARY")
    print("=" * 50)
    print(f"Total PDFs found:     {summary['total_files']}")
    print(f"Successfully processed: {summary['successful']}")
    print(f"Failed:                {summary['failed']}")
    print(f"Total chunks created:  {summary['total_chunks']}")
    print(f"Database entries:      {summary['total_entries']}")

    if summary["errors"]:
        print("\nErrors:")
        for err in summary["errors"]:
            print(f"  - {err['file']}: {err['error']}")

    return summary


if __name__ == "__main__":
    main()
